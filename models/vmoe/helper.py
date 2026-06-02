import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augment import build_class_prototypes, prototype_nearest_neighbor_augment
from utils import Averager, count_acc


def amp_autocast(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def labels_to_session_ids(labels, args):
    session_ids = labels.new_zeros(labels.shape)
    mask = labels >= args.base_class
    if mask.any():
        session_ids[mask] = 1 + (labels[mask] - args.base_class) // args.way
    return session_ids.long()


def _build_prev_masks(args, session):
    prev_masks = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for prev_session in range(session):
        if prev_session == 0:
            prev_masks.append(torch.arange(args.base_class, device=device))
        else:
            offset = args.base_class + (prev_session - 1) * args.way
            prev_masks.append(torch.arange(offset, offset + args.way, device=device))
    return prev_masks


def _set_stage_mode(model, stage, train):
    if stage == 'visual':
        model.train() if train else model.eval()
        model.module.router.eval()
        return

    model.module.backbone.eval()
    model.module.classifier.eval()
    if not train:
        model.module.router.eval()
        return

    if getattr(model.module.router.feature_extractor, 'uses_warp', False):
        model.module.router.eval()
        model.module.router.discriminator.train()
    else:
        model.module.router.train()


def run_visual_epoch(model, loader, optimizer, scheduler, epoch, args, session, train=True, scaler=None):
    loss_meter = Averager()
    acc_meter = Averager()
    _set_stage_mode(model, 'visual', train)
    context = torch.enable_grad() if train else torch.no_grad()
    ortho_weight = args.lambda_ortho_base if session == 0 else args.lambda_ortho_new

    with context:
        iterator = tqdm(loader, leave=False)
        for batch in iterator:
            images, labels = [_.cuda(non_blocking=True) for _ in batch]
            with amp_autocast(enabled=getattr(args, 'use_amp', False)):
                features = model.module.encode(images, session)
                logits = model.module.forward_with_features(features, session)
                logits = logits[:, :model.module.seen_classes(session)]
                ce_loss = F.cross_entropy(logits, labels)
                ortho_loss = model.module.classifier.orthogonality_loss(
                    session,
                    model.module.get_session_class_mask(session),
                    _build_prev_masks(args, session),
                    lambda_cross=args.lambda_cross,
                )
                loss = ce_loss + ortho_weight * ortho_loss
            acc = count_acc(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if getattr(args, 'use_amp', False) and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            tag = 'visual-train' if train else 'visual-eval'
            iterator.set_description(f'{tag} s{session} e{epoch} loss={loss.item():.4f} acc={acc:.4f}')
            loss_meter.add(loss.item())
            acc_meter.add(acc)

    if train and scheduler is not None:
        scheduler.step()
    return loss_meter.item(), acc_meter.item()


def eval_visual(model, loader, args, session):
    loss_meter = Averager()
    acc_meter = Averager()
    _set_stage_mode(model, 'visual', train=False)

    with torch.no_grad():
        iterator = tqdm(loader, leave=False)
        for batch in iterator:
            images, labels = [_.cuda(non_blocking=True) for _ in batch]
            session_ids = labels_to_session_ids(labels, args)
            with amp_autocast(enabled=getattr(args, 'use_amp', False)):
                logits = model(images, session=session_ids)
                logits = logits[:, :model.module.seen_classes(session)]
                loss = F.cross_entropy(logits, labels)
            acc = count_acc(logits, labels)
            iterator.set_description(f'visual-eval s{session} loss={loss.item():.4f} acc={acc:.4f}')
            loss_meter.add(loss.item())
            acc_meter.add(acc)
    return loss_meter.item(), acc_meter.item()


def run_router_epoch(model, loader, optimizer, scheduler, epoch, args, session, train=True, route_prototypes=None, scaler=None):
    loss_meter = Averager()
    acc_meter = Averager()
    _set_stage_mode(model, 'router', train)
    context = torch.enable_grad() if train else torch.no_grad()
    class_ids = model.module.get_session_class_ids(session)

    with context:
        iterator = tqdm(loader, leave=False)
        for batch in iterator:
            images, labels = [_.cuda(non_blocking=True) for _ in batch]
            session_targets = labels_to_session_ids(labels, args)
            with amp_autocast(enabled=getattr(args, 'use_amp', False)):
                _, route_features = model.module.router.extract_route_features(images)

                if route_prototypes is None:
                    effective_route_prototypes = build_class_prototypes(route_features.detach(), labels, class_ids)
                else:
                    effective_route_prototypes = route_prototypes

                if args.router_disc_type == 'msd':
                    if session > 0:
                        aug_route_features, aug_labels = prototype_nearest_neighbor_augment(
                            route_features,
                            labels,
                            effective_route_prototypes,
                            k=args.router_neighbor_k,
                            sample_n=args.router_sample_n,
                            lambda_min=args.aug_lambda_min,
                            lambda_max=args.aug_lambda_max,
                        )
                        route_targets = labels_to_session_ids(aug_labels, args)
                    else:
                        aug_route_features = route_features
                        route_targets = session_targets
                    route_logits = model.module.router.discriminator(
                        aug_route_features,
                        session,
                        session_memory_overrides={session: effective_route_prototypes},
                    )
                    loss = F.cross_entropy(route_logits, route_targets)
                    acc = count_acc(route_logits, route_targets)
                else:
                    errors = model.module.router.discriminator(route_features, session)
                    loss = F.cross_entropy(-errors, session_targets)
                    acc = count_acc(-errors, session_targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if getattr(args, 'use_amp', False) and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            tag = 'router-train' if train else 'router-eval'
            iterator.set_description(f'{tag} s{session} e{epoch} loss={loss.item():.4f} acc={acc:.4f}')
            loss_meter.add(loss.item())
            acc_meter.add(acc)

    if train and scheduler is not None:
        scheduler.step()
    return loss_meter.item(), acc_meter.item()


def eval_router(model, loader, args, session):
    loss_meter = Averager()
    acc_meter = Averager()
    _set_stage_mode(model, 'router', train=False)

    with torch.no_grad():
        iterator = tqdm(loader, leave=False)
        for batch in iterator:
            images, labels = [_.cuda(non_blocking=True) for _ in batch]
            session_targets = labels_to_session_ids(labels, args)
            with amp_autocast(enabled=getattr(args, 'use_amp', False)):
                _, route_features = model.module.router.extract_route_features(images)
                if args.router_disc_type == 'msd':
                    route_logits = model.module.router.discriminator(route_features, session)
                    loss = F.cross_entropy(route_logits, session_targets)
                    acc = count_acc(route_logits, session_targets)
                else:
                    errors = model.module.router.discriminator(route_features, session)
                    loss = F.cross_entropy(-errors, session_targets)
                    acc = count_acc(-errors, session_targets)
            iterator.set_description(f'router-eval s{session} loss={loss.item():.4f} acc={acc:.4f}')
            loss_meter.add(loss.item())
            acc_meter.add(acc)
    return loss_meter.item(), acc_meter.item()


def eval_overall(model, loader, args, session):
    loss_meter = Averager()
    acc_meter = Averager()
    model.eval()
    model.module.backbone.eval()
    model.module.classifier.eval()
    model.module.router.eval()

    with torch.no_grad():
        iterator = tqdm(loader, leave=False)
        for batch in iterator:
            images, labels = [_.cuda(non_blocking=True) for _ in batch]
            with amp_autocast(enabled=getattr(args, 'use_amp', False)):
                logits = model(images)
                logits = logits[:, :model.module.seen_classes(session)]
                loss = F.cross_entropy(logits, labels)
            acc = count_acc(logits, labels)
            iterator.set_description(f'overall-eval s{session} loss={loss.item():.4f} acc={acc:.4f}')
            loss_meter.add(loss.item())
            acc_meter.add(acc)
    return loss_meter.item(), acc_meter.item()
