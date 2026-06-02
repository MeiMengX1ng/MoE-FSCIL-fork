import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .warp import compute_warp_orthonormal_basis, identify_warp_importance, restore_warp_weights, switch_warp_modules, WaRPModule


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class FrozenResNet18Extractor(nn.Module):
    uses_warp = False

    def __init__(self, args):
        super().__init__()
        self.image_size = args.model_image_size
        self.encoder = ResNet18FeatureExtractor(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return self.encoder(x)

    def trainable_parameters(self):
        return []


class WaRPResNet18Extractor(nn.Module):
    uses_warp = True

    def __init__(self, args):
        super().__init__()
        self.image_size = args.model_image_size
        base_encoder = ResNet18FeatureExtractor(pretrained=True)
        self.encoder = switch_warp_modules(base_encoder)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for module in self.encoder.modules():
            if isinstance(module, WaRPModule):
                module.basis_coeff.requires_grad = True

    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return self.encoder(x)

    def trainable_parameters(self):
        params = []
        for module in self.encoder.modules():
            if isinstance(module, WaRPModule):
                params.append(module.basis_coeff)
        return params

    def compute_basis(self, dataloader, max_batches=None):
        compute_warp_orthonormal_basis(self.encoder, dataloader, max_batches=max_batches)

    def identify_importance(self, dataloader, loss_closure, keep_ratio, max_batches=None, zero_grad_fn=None):
        return identify_warp_importance(
            self.encoder,
            dataloader,
            loss_closure,
            keep_ratio=keep_ratio,
            max_batches=max_batches,
            zero_grad_fn=zero_grad_fn,
        )

    def restore_weights(self):
        restore_warp_weights(self.encoder)


class MemoryBankSessionDiscriminator(nn.Module):
    def __init__(self, feat_dim, max_sessions):
        super().__init__()
        self.max_sessions = max_sessions
        self.projector = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
        )
        self.memory_bank = {}

    def update_memory(self, session_id, class_prototypes):
        self.memory_bank[int(session_id)] = {int(k): v.detach().clone() for k, v in class_prototypes.items()}

    def forward(self, features, active_sessions, session_memory_overrides=None):
        session_memory_overrides = session_memory_overrides or {}
        pooled = []
        for session_id in range(active_sessions + 1):
            session_memory = session_memory_overrides.get(session_id, self.memory_bank.get(session_id, {}))
            if session_memory:
                pooled.append(torch.stack(list(session_memory.values()), dim=0).mean(dim=0))
            else:
                pooled.append(features.new_zeros(features.size(-1)))
        pooled = torch.stack(pooled, dim=0)
        pooled = pooled.unsqueeze(0).expand(features.size(0), -1, -1)
        features = features.unsqueeze(1).expand_as(pooled)
        fused = torch.cat([features, pooled], dim=-1)
        logits = self.projector(fused).squeeze(-1)
        return logits[:, :active_sessions + 1]


class SessionAutoEncoder(nn.Module):
    def __init__(self, feat_dim, bottleneck_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim // 2, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, feat_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DistributionSessionDiscriminator(nn.Module):
    def __init__(self, feat_dim, max_sessions, bottleneck_dim=256):
        super().__init__()
        self.autoencoders = nn.ModuleList(
            SessionAutoEncoder(feat_dim, bottleneck_dim=bottleneck_dim)
            for _ in range(max_sessions)
        )

    def forward(self, features, active_sessions):
        errors = []
        for session_id in range(active_sessions + 1):
            recon = self.autoencoders[session_id](features)
            errors.append((features - recon).pow(2).mean(dim=-1))
        return torch.stack(errors, dim=-1)

    def predict_session(self, features, active_sessions):
        errors = self.forward(features, active_sessions)
        return torch.argmin(errors, dim=-1)


class SessionRoutingModule(nn.Module):
    def __init__(self, args, max_sessions):
        super().__init__()
        self.args = args
        self.max_sessions = max_sessions
        if args.router_feat_mode == 'warp':
            self.feature_extractor = WaRPResNet18Extractor(args)
        else:
            self.feature_extractor = FrozenResNet18Extractor(args)

        self.route_feat_dim = 512
        if args.router_disc_type == 'dsd':
            self.discriminator = DistributionSessionDiscriminator(
                feat_dim=self.route_feat_dim,
                max_sessions=max_sessions,
                bottleneck_dim=args.router_bottleneck_dim,
            )
        else:
            self.discriminator = MemoryBankSessionDiscriminator(
                feat_dim=self.route_feat_dim,
                max_sessions=max_sessions,
            )

    def extract_route_features(self, x):
        feat_map = self.feature_extractor(x)
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return feat_map, pooled

    def forward(self, x, active_sessions):
        _, pooled = self.extract_route_features(x)
        if self.args.router_disc_type == 'dsd':
            return self.discriminator.predict_session(pooled, active_sessions)
        logits = self.discriminator(pooled, active_sessions)
        return torch.argmax(logits, dim=-1)
