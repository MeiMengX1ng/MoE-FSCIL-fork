import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vit_b_16
except Exception:  # pragma: no cover
    vit_b_16 = None

from models.resnet18_encoder import resnet18
from .classifier import SessionDecoupledClassifier
from .lora import MultiSessionLoRA, LoRALinear, inject_lora_to_linear_layers
from .router import SessionRoutingModule


class FrozenVisionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.model_image_size
        self.out_dim = args.backbone_feat_dim
        self.kind = 'resnet'
        self.lora_linear_layers = []

        if args.backbone_type == 'clip_vit_b16' and vit_b_16 is not None:
            vit = vit_b_16(weights=None)
            self.backbone = vit
            self.out_dim = 768
            self.kind = 'vit'
            self.lora_linear_layers = inject_lora_to_linear_layers(
                self.backbone.encoder.layers,
                num_sessions=args.sessions,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                exclude_names={'head', 'heads'},
            )
        else:
            self.backbone = resnet18(False, args)
            self.out_dim = 512

        if args.backbone_model_dir is not None:
            state = torch.load(args.backbone_model_dir)
            state = state.get('params', state)
            self.backbone.load_state_dict(state, strict=False)

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
        if self.kind == 'vit':
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)
            return x[:, 0]

        feat = self.backbone(x)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat


class VMOENet(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.args = args
        self.mode = mode or args.base_mode
        self.backbone = FrozenVisionBackbone(args)
        self.num_features = self.backbone.out_dim
        self.use_backbone_lora = self.backbone.kind == 'vit'
        self.lora = None
        if not self.use_backbone_lora:
            self.lora = MultiSessionLoRA(
                self.num_features,
                self.num_features,
                num_sessions=args.sessions,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )
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
        if self.use_backbone_lora:
            self.backbone.set_active_session(session)
        base_feat = self.backbone(x)
        if session is None or self.use_backbone_lora:
            return base_feat
        return base_feat + self.lora(base_feat, session)

    def forward_with_features(self, features, session):
        return self.classifier(features, session, self.args.base_class, self.args.way)

    def _forward_mixed_sessions(self, x, session_ids):
        session_ids = session_ids.to(x.device)
        logits = x.new_full((x.size(0), self.args.num_classes), float('-inf'))
        for session in torch.unique(session_ids).tolist():
            mask = session_ids == int(session)
            features = self.encode(x[mask], session=int(session))
            session_logits = self.forward_with_features(features, int(session))
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
