import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionLoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        update = F.linear(F.linear(x, self.A), self.B)
        scale = self.alpha / max(self.rank, 1)
        return update * scale


class MultiSessionLoRA(nn.Module):
    def __init__(self, in_features, out_features, num_sessions, rank=8, alpha=1.0):
        super().__init__()
        self.num_sessions = num_sessions
        self.experts = nn.ModuleList(
            SessionLoRAExpert(in_features, out_features, rank=rank, alpha=alpha)
            for _ in range(num_sessions)
        )

    def forward(self, x, session_id):
        if session_id is None:
            return torch.zeros_like(x)
        session_id = int(session_id)
        if session_id < 0 or session_id >= self.num_sessions:
            raise ValueError(f"session_id {session_id} is out of range")
        return self.experts[session_id](x)

    def trainable_parameters_for_session(self, session_id):
        return list(self.experts[int(session_id)].parameters())


class LoRALinear(nn.Module):
    def __init__(self, linear, num_sessions, rank=8, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.num_sessions = num_sessions
        self.rank = rank
        self.alpha = alpha
        self.active_session = None
        for param in self.linear.parameters():
            param.requires_grad = False
        self.session_lora = MultiSessionLoRA(
            in_features=linear.in_features,
            out_features=linear.out_features,
            num_sessions=num_sessions,
            rank=rank,
            alpha=alpha,
        )

    def set_active_session(self, session_id):
        self.active_session = None if session_id is None else int(session_id)

    def forward(self, x):
        base = self.linear(x)
        if self.active_session is None:
            return base
        return base + self.session_lora(x, self.active_session)

    def trainable_parameters_for_session(self, session_id):
        return self.session_lora.trainable_parameters_for_session(session_id)


def inject_lora_to_linear_layers(module, num_sessions, rank=8, alpha=1.0, exclude_names=None):
    exclude_names = set() if exclude_names is None else set(exclude_names)
    wrapped = []
    for name, child in list(module.named_children()):
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            lora_child = LoRALinear(child, num_sessions=num_sessions, rank=rank, alpha=alpha)
            setattr(module, name, lora_child)
            wrapped.append(lora_child)
        else:
            wrapped.extend(
                inject_lora_to_linear_layers(
                    child,
                    num_sessions=num_sessions,
                    rank=rank,
                    alpha=alpha,
                    exclude_names=exclude_names,
                )
            )
    return wrapped


def inject_lora_to_vit_encoder_layers(encoder_layers, num_sessions, rank=8, alpha=1.0):
    wrapped = []
    for _, block in encoder_layers.named_children():
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1') and hasattr(block.mlp, 'fc2'):
            if isinstance(block.mlp.fc1, nn.Linear):
                lora_fc1 = LoRALinear(block.mlp.fc1, num_sessions=num_sessions, rank=rank, alpha=alpha)
                block.mlp.fc1 = lora_fc1
                wrapped.append(lora_fc1)
            if isinstance(block.mlp.fc2, nn.Linear):
                lora_fc2 = LoRALinear(block.mlp.fc2, num_sessions=num_sessions, rank=rank, alpha=alpha)
                block.mlp.fc2 = lora_fc2
                wrapped.append(lora_fc2)
    return wrapped
