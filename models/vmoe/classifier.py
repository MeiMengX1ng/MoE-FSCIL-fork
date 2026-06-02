import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionDecoupledClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, sessions, temperature=16.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.sessions = sessions
        self.temperature = temperature
        self.heads = nn.ModuleList(
            nn.Linear(feat_dim, num_classes, bias=False) for _ in range(sessions)
        )

    def build_weight_matrix(self, upto_session, base_class, way):
        weight = self.heads[0].weight[:base_class]
        parts = [weight]
        for session in range(1, int(upto_session) + 1):
            start = base_class + (session - 1) * way
            end = start + way
            parts.append(self.heads[session].weight[start:end])
        return torch.cat(parts, dim=0)

    def forward(self, features, upto_session, base_class, way):
        weight = self.build_weight_matrix(upto_session, base_class, way)
        return self.temperature * F.linear(F.normalize(features, p=2, dim=-1), F.normalize(weight, p=2, dim=-1))

    def get_session_weights(self, session_id, class_indices):
        weight = self.heads[int(session_id)].weight[class_indices]
        return weight

    def orthogonality_loss(self, session_id, class_mask, prev_class_mask=None, lambda_cross=0.8):
        session_id = int(session_id)
        current_weight = self.heads[session_id].weight[class_mask]
        if current_weight.numel() == 0:
            return current_weight.new_tensor(0.0)

        current_weight = F.normalize(current_weight, p=2, dim=-1)
        gram = current_weight @ current_weight.t()
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        intra = (gram - eye).pow(2).mean()

        if prev_class_mask is None or session_id == 0:
            return intra

        prev_weight = []
        for idx in range(session_id):
            prev_weight.append(self.heads[idx].weight[prev_class_mask[idx]])
        prev_weight = [w for w in prev_weight if w.numel() > 0]
        if not prev_weight:
            return intra

        prev_weight = F.normalize(torch.cat(prev_weight, dim=0), p=2, dim=-1)
        cross = (current_weight @ prev_weight.t()).pow(2).mean()
        return intra + lambda_cross * cross
