import torch
import torch.nn.functional as F


def build_class_prototypes(features, labels, class_ids):
    prototypes = {}
    for class_id in class_ids:
        mask = labels == class_id
        if mask.any():
            prototypes[int(class_id)] = features[mask].mean(dim=0)
    return prototypes


def prototype_nearest_neighbor_augment(features, labels, prototypes, k=5, sample_n=5, lambda_min=0.45, lambda_max=0.75):
    if not prototypes:
        return features, labels

    class_ids = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[class_id] for class_id in class_ids], dim=0)
    proto_matrix = F.normalize(proto_matrix, p=2, dim=-1)

    aug_features = [features]
    aug_labels = [labels]
    for _ in range(sample_n):
        shifted = []
        shifted_labels = []
        for feature, label in zip(features, labels):
            label = int(label.item())
            if label not in prototypes:
                continue
            center = prototypes[label]
            sims = F.linear(F.normalize(center.unsqueeze(0), p=2, dim=-1), proto_matrix).squeeze(0)
            ranked = torch.argsort(sims, descending=True)
            neighbors = [class_ids[idx.item()] for idx in ranked if class_ids[idx.item()] != label][:k]
            for neighbor_id in neighbors:
                neighbor = prototypes[neighbor_id]
                lam = torch.empty(1, device=feature.device).uniform_(lambda_min, lambda_max).item()
                delta = 0.5 * (center - neighbor)
                shifted.append(feature + lam * delta)
                shifted_labels.append(label)
        if shifted:
            aug_features.append(torch.stack(shifted, dim=0))
            aug_labels.append(labels.new_tensor(shifted_labels))

    return torch.cat(aug_features, dim=0), torch.cat(aug_labels, dim=0)

