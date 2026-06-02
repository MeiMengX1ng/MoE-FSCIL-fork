import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import SessionDecoupledClassifier
from .clip_loader import get_clip_encoder_layers, load_clip_vision_backbone
from .lora import inject_lora_to_vit_encoder_layers
from .router import SessionRoutingModule


class FrozenVisionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.model_image_size
        self.out_dim = 768
        self.kind = 'vit'
        self.lora_linear_layers = []

        if args.backbone_type != 'clip_vit_b16':
            raise ValueError("VMOE currently supports only clip_vit_b16 as the classification backbone")

        self.backbone = load_clip_vision_backbone()
        self.lora_linear_layers = inject_lora_to_vit_encoder_layers(
            get_clip_encoder_layers(self.backbone),
            num_sessions=args.sessions,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.lora_linear_layers:
            for param in module.session_lora.parameters():
                param.requires_grad = True

    def set_active_session(self, session):
        for module in self.lora_linear_layers:
            module.set_active_session(session)

    def trainable_parameters_for_session(self, session):
        params = []
        for module in self.lora_linear_layers:
            params.extend(module.trainable_parameters_for_session(session))
        return params

    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        outputs = self.backbone(pixel_values=x)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]


class VMOENet(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode or args.base_mode
        self.backbone = FrozenVisionBackbone(args)
        self.num_features = self.backbone.out_dim
        self.use_backbone_lora = True
        self.classifier = SessionDecoupledClassifier(
            feat_dim=self.num_features,
            num_classes=args.num_classes,
            sessions=args.sessions,
            temperature=args.temperature,
        )
        self.router = SessionRoutingModule(args, max_sessions=args.sessions)
        self.active_session = 0

    def seen_classes(self, session):
        return self.args.base_class + session * self.args.way

    def get_session_class_ids(self, session):
        if session == 0:
            return list(range(self.args.base_class))
        start = self.args.base_class + (session - 1) * self.args.way
        return list(range(start, start + self.args.way))

    def get_session_class_mask(self, session):
        return torch.tensor(self.get_session_class_ids(session), dtype=torch.long, device=self.classifier.heads[0].weight.device)

    def encode(self, x, session=None):
        self.backbone.set_active_session(session)
        base_feat = self.backbone(x)
        return base_feat

    def forward_with_features(self, features, session):
        return self.classifier(features, session, self.args.base_class, self.args.way)

    def _forward_mixed_sessions(self, x, session_ids):
        session_ids = session_ids.to(x.device)
        logits = None
        for session in torch.unique(session_ids).tolist():
            mask = session_ids == int(session)
            features = self.encode(x[mask], session=int(session))
            session_logits = self.forward_with_features(features, int(session))
            if logits is None:
                logits = session_logits.new_full((x.size(0), self.args.num_classes), float('-inf'))
            logits[mask, :session_logits.size(1)] = session_logits
        return logits

    def infer_session(self, x, active_sessions):
        pred = self.router(x, active_sessions)
        if pred.ndim == 0:
            pred = pred.unsqueeze(0)
        return pred

    def forward(self, x, session=None):
        if self.mode == 'encoder':
            return self.encode(x, session=session)

        if session is None:
            session = self.infer_session(x, self.active_session)

        if torch.is_tensor(session):
            if session.ndim == 0:
                session = int(session.item())
            else:
                return self._forward_mixed_sessions(x, session)

        features = self.encode(x, session=int(session))
        return self.forward_with_features(features, int(session))
