import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augment import build_class_prototypes, prototype_nearest_neighbor_augment
from utils import Averager, count_acc


def _build_prev_masks(args, session):
    prev_masks = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = args.base_class
    for prev_session in range(session):
        if prev_session == 0:
            prev_masks.append(torch.arange(args.base_class, device=device))
        else:
            offset = args.base_class + (prev_session - 1) * args.way
            prev_masks.append(torch.arange(offset, offset + args.way, device=device))
    return prev_masks


def run_epoch(model, loader, optimizer, scheduler, epoch, args, session, train=True):
    loss_meter = Averager()
    acc_meter = Averager()

    model.train() if train else model.eval()
    context = torch.enable_grad() if train else torch.no_grad()
    class_ids = model.module.get_session_class_ids(session)

    ortho_weight = args.lambda_ortho_base if session == 0 else args.lambda_ortho_new

    with context:
        iterator = tqdm(loader)
        for batch in iterator:
            images, labels = [_.cuda() for _ in batch]
            features = model.module.encode(images, session)
            _, route_features = model.module.router.extract_route_features(images)

            route_prototypes = build_class_prototypes(route_features.detach(), labels, class_ids)
            aug_route_features, aug_labels = prototype_nearest_neighbor_augment(
                route_features,
                labels,
                route_prototypes,
                k=args.router_neighbor_k,
                sample_n=args.router_sample_n,
                lambda_min=args.aug_lambda_min,
                lambda_max=args.aug_lambda_max,
            )

            logits = model.module.forward_with_features(features, session)
            ce_loss = F.cross_entropy(logits[:, :model.module.seen_classes(session)], labels)
            ortho_loss = model.module.classifier.orthogonality_loss(
                session,
                model.module.get_session_class_mask(session),
                _build_prev_masks(args, session),
                lambda_cross=args.lambda_cross,
            )
            if args.router_disc_type == 'msd':
                route_logits = model.module.router.discriminator(aug_route_features.detach(), session)
                route_target = labels.new_full((aug_route_features.size(0),), session)
                route_loss = F.cross_entropy(route_logits, route_target)
            else:
                recon = model.module.router.discriminator.autoencoders[session](aug_route_features.detach())
                route_loss = F.mse_loss(recon, aug_route_features.detach())

            loss = ce_loss + ortho_weight * ortho_loss + args.router_loss_weight * route_loss
            acc = count_acc(logits[:, :model.module.seen_classes(session)], labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iterator.set_description(
                f"Session {session}, epoch {epoch}, loss={loss.item():.4f}, acc={acc:.4f}"
            )
            loss_meter.add(loss.item())
            acc_meter.add(acc)

    if train and scheduler is not None:
        scheduler.step()
    return loss_meter.item(), acc_meter.item()


def eval_model(model, loader, args, session):
    model.eval()
    loss_meter = Averager()
    acc_meter = Averager()
    with torch.no_grad():
        iterator = tqdm(loader)
        for batch in iterator:
            images, labels = [_.cuda() for _ in batch]
            logits = model(images, session=None)
            logits = logits[:, :model.module.seen_classes(session)]
            loss = F.cross_entropy(logits, labels)
            acc = count_acc(logits, labels)
            loss_meter.add(loss.item())
            acc_meter.add(acc)
    return loss_meter.item(), acc_meter.item()
