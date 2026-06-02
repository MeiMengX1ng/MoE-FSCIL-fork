import json
import os
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn

from dataloader.data_utils import get_base_dataloader, get_new_dataloader, set_up_datasets
from models.base.base import Trainer
from utils import ensure_path, save_list_to_txt
from .Network import VMOENet
from .augment import build_class_prototypes
from .helper import eval_overall, eval_router, eval_visual, labels_to_session_ids, run_router_epoch, run_visual_epoch


def build_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_checkpoint_path(model_dir, start_session):
    path = Path(model_dir).expanduser()
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {model_dir}")

    candidates = []
    if start_session > 0:
        previous_session = start_session - 1
        candidates.extend([
            path / f"session{previous_session}_last.pth",
            path / f"session{previous_session}_router_best.pth",
            path / f"session{previous_session}_visual_best.pth",
        ])
    else:
        for pattern in ('session*_last.pth', 'session*_router_best.pth', 'session*_visual_best.pth'):
            candidates.extend(sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"No compatible checkpoint found in directory: {model_dir}. "
        "Expected session*_last.pth, session*_router_best.pth, or session*_visual_best.pth"
    )


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = set_up_datasets(args)
        self.args.use_amp = torch.cuda.is_available()
        self.scaler = build_grad_scaler(enabled=self.args.use_amp)
        self.visual_max_acc = [0.0] * self.args.sessions
        self.router_max_acc = [0.0] * self.args.sessions
        self.overall_max_acc = [0.0] * self.args.sessions
        self.training_log = []
        self.result_log = [self.serializable_args()]
        self.set_save_path()
        self.save_config()
        self.model = nn.DataParallel(VMOENet(self.args, mode=self.args.base_mode), list(range(self.args.num_gpu))).cuda()
        if self.args.model_dir is not None:
            checkpoint_path = resolve_checkpoint_path(self.args.model_dir, self.args.start_session)
            print(f"Loading local checkpoint: {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['params'] if isinstance(checkpoint, dict) and 'params' in checkpoint else checkpoint
            extra_state = checkpoint.get('extra_state', {}) if isinstance(checkpoint, dict) else {}
            metrics_state = checkpoint.get('metrics_state', {}) if isinstance(checkpoint, dict) else {}
            self.best_model_dict = state_dict
            self.model.load_state_dict(self.best_model_dict, strict=False)
            self._load_extra_state(extra_state)
            self._load_metrics_state(metrics_state)
            self.training_log = checkpoint.get('training_log', []) if isinstance(checkpoint, dict) else []
            self.result_log = checkpoint.get('result_log', [self.serializable_args()]) if isinstance(checkpoint, dict) else [self.serializable_args()]
            self.persist_training_log(snapshot_only=True)
        else:
            self.best_model_dict = deepcopy(self.model.state_dict())

    def set_save_path(self):
        checkpoint_root = '/data/xuzg/FSCIL/MoE-FSCIL/checkpoint'
        router_tag = f"{self.args.router_disc_type}-{self.args.router_feat_mode}-router2stage"
        lora_tag = f"lora_r{self.args.lora_rank}_a{self.args.lora_alpha:.1f}"
        ortho_tag = (
            f"ortho_b{self.args.lambda_ortho_base:g}_"
            f"n{self.args.lambda_ortho_new:g}_"
            f"cross{self.args.lambda_cross:g}"
        )
        train_tag = (
            f"visB{self.args.epochs_base}_visN{self.args.epochs_new}_"
            f"rtB{self.args.router_epochs_base}_rtN{self.args.router_epochs_new}_"
            f"lrB{self.args.lr_base:g}_lrN{self.args.lr_new:g}_seed{self.args.seed}"
        )
        if self.args.router_feat_mode == 'warp':
            train_tag = f"{train_tag}_keep{self.args.fraction_to_keep:g}"
        run_name = f"{router_tag}-{lora_tag}-{ortho_tag}-{train_tag}"
        self.args.save_path = os.path.join(checkpoint_root, self.args.dataset, self.args.project, run_name)
        ensure_path(self.args.save_path)

    def serializable_args(self) -> Dict[str, Any]:
        config = {}
        for key, value in vars(self.args).items():
            try:
                json.dumps(value)
                config[key] = value
            except TypeError:
                config[key] = getattr(value, '__name__', str(value))
        return config

    def save_config(self):
        config_path = os.path.join(self.args.save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.serializable_args(), f, indent=2, sort_keys=True)

    def _jsonl_path(self):
        return os.path.join(self.args.save_path, 'training_log.jsonl')

    def persist_training_log(self, snapshot_only=False):
        json_path = os.path.join(self.args.save_path, 'training_log.json')
        with open(json_path, 'w') as f:
            json.dump(self.training_log, f, indent=2, sort_keys=True)
        if snapshot_only:
            with open(self._jsonl_path(), 'w') as f:
                for record in self.training_log:
                    f.write(json.dumps(record, sort_keys=True) + '\n')

    def log_event(self, stage, session, **metrics):
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'stage': stage,
            'session': int(session),
        }
        record.update(metrics)
        self.training_log.append(record)
        with open(self._jsonl_path(), 'a') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        self.persist_training_log(snapshot_only=False)
        return record

    def _metrics_state(self):
        return {
            'visual_max_acc': self.visual_max_acc,
            'router_max_acc': self.router_max_acc,
            'overall_max_acc': self.overall_max_acc,
            'trlog': self.trlog,
        }

    def _load_metrics_state(self, metrics_state):
        self.visual_max_acc = metrics_state.get('visual_max_acc', self.visual_max_acc)
        self.router_max_acc = metrics_state.get('router_max_acc', self.router_max_acc)
        self.overall_max_acc = metrics_state.get('overall_max_acc', self.overall_max_acc)
        saved_trlog = metrics_state.get('trlog')
        if saved_trlog is not None:
            self.trlog = saved_trlog

    def _extra_state(self):
        state = {'active_session': self.model.module.active_session}
        if self.args.router_disc_type == 'msd':
            memory_bank = {}
            for session_id, class_bank in self.model.module.router.discriminator.memory_bank.items():
                memory_bank[int(session_id)] = {
                    int(class_id): tensor.detach().cpu()
                    for class_id, tensor in class_bank.items()
                }
            state['memory_bank'] = memory_bank
        return state

    def _load_extra_state(self, extra_state):
        self.model.module.active_session = extra_state.get('active_session', self.args.start_session)
        if self.args.router_disc_type == 'msd':
            restored = {}
            for session_id, class_bank in extra_state.get('memory_bank', {}).items():
                restored[int(session_id)] = {
                    int(class_id): tensor.cuda()
                    for class_id, tensor in class_bank.items()
                }
            self.model.module.router.discriminator.memory_bank = restored

    def save_checkpoint(self, path):
        torch.save(
            {
                'params': self.model.state_dict(),
                'extra_state': self._extra_state(),
                'metrics_state': self._metrics_state(),
                'training_log': self.training_log,
                'result_log': self.result_log,
            },
            path,
        )

    def get_dataloader(self, session):
        if session == 0:
            return get_base_dataloader(self.args)
        return get_new_dataloader(self.args, session)

    def _epoch_count(self, session, stage):
        if stage == 'visual':
            return self.args.epochs_base if session == 0 else self.args.epochs_new
        return self.args.router_epochs_base if session == 0 else self.args.router_epochs_new

    def _lr_for_session(self, session):
        return self.args.lr_base if session == 0 else self.args.lr_new

    def _schedule_for_session(self, session):
        return self.args.milestones if session == 0 else self.args.milestones_new

    def get_visual_optimizer(self, session):
        params = []
        params.extend(self.model.module.backbone.trainable_parameters_for_session(session))
        params.extend(self.model.module.classifier.heads[session].parameters())
        optimizer = torch.optim.SGD(
            params,
            lr=self._lr_for_session(session),
            momentum=self.args.momentum,
            weight_decay=self.args.decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self._schedule_for_session(session),
            gamma=self.args.gamma,
        )
        return optimizer, scheduler

    def get_router_optimizer(self, session):
        if getattr(self.model.module.router.feature_extractor, 'uses_warp', False):
            params = list(self.model.module.router.feature_extractor.trainable_parameters())
        else:
            for param in self.model.module.router.feature_extractor.parameters():
                param.requires_grad = True
            params = list(self.model.module.router.feature_extractor.parameters())

        if self.args.router_disc_type == 'msd':
            params.extend(self.model.module.router.discriminator.projector.parameters())
        else:
            params.extend(self.model.module.router.discriminator.autoencoders[session].parameters())

        optimizer = torch.optim.SGD(
            params,
            lr=self._lr_for_session(session),
            momentum=self.args.momentum,
            weight_decay=self.args.decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self._schedule_for_session(session),
            gamma=self.args.gamma,
        )
        return optimizer, scheduler

    def _router_warp_importance_loss(self, images, labels, session, route_prototypes):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        session_targets = labels_to_session_ids(labels, self.args)
        _, route_features = self.model.module.router.extract_route_features(images)
        if self.args.router_disc_type == 'msd':
            route_logits = self.model.module.router.discriminator(
                route_features,
                session,
                session_memory_overrides={session: route_prototypes},
            )
            return torch.nn.functional.cross_entropy(route_logits, session_targets)

        errors = self.model.module.router.discriminator(route_features, session)
        return torch.nn.functional.cross_entropy(-errors, session_targets)

    def update_router_warp_mask(self, trainloader, session, route_prototypes):
        extractor = self.model.module.router.feature_extractor
        if not getattr(extractor, 'uses_warp', False):
            return None

        self.model.module.backbone.eval()
        self.model.module.classifier.eval()
        self.model.module.router.eval()
        self.model.module.router.discriminator.train()

        keep_ratio = extractor.identify_importance(
            trainloader,
            loss_closure=lambda images, labels: self._router_warp_importance_loss(images, labels, session, route_prototypes),
            keep_ratio=self.args.fraction_to_keep,
            max_batches=self.args.warp_importance_max_batches,
            zero_grad_fn=lambda: self.model.zero_grad(set_to_none=True),
        )
        self.model.zero_grad(set_to_none=True)
        return keep_ratio

    def collect_route_prototypes(self, trainloader, session):
        feats = []
        labels = []
        self.model.module.backbone.eval()
        self.model.module.classifier.eval()
        self.model.module.router.eval()
        with torch.no_grad():
            for batch in trainloader:
                images, target = [_.cuda(non_blocking=True) for _ in batch]
                _, pooled = self.model.module.router.extract_route_features(images)
                feats.append(pooled)
                labels.append(target)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return build_class_prototypes(feats, labels, self.model.module.get_session_class_ids(session))

    def update_router_memory(self, trainloader, session, prototypes=None):
        if self.args.router_disc_type != 'msd':
            return None
        prototypes = prototypes or self.collect_route_prototypes(trainloader, session)
        self.model.module.router.discriminator.update_memory(session, prototypes)
        return prototypes

    def run_visual_stage(self, session, trainloader, testloader, result_list):
        optimizer, scheduler = self.get_visual_optimizer(session)
        epochs = self._epoch_count(session, 'visual')
        best_acc = -1.0
        best_state = deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            train_loss, train_acc = run_visual_epoch(
                self.model,
                trainloader,
                optimizer,
                scheduler,
                epoch,
                self.args,
                session,
                train=True,
                scaler=self.scaler,
            )
            test_loss, test_acc = eval_visual(self.model, testloader, self.args, session)
            print(
                f'[visual] session={session} epoch={epoch} '
                f'train_loss={train_loss:.5f} train_acc={train_acc:.5f} '
                f'test_loss={test_loss:.5f} test_acc={test_acc:.5f}',
                flush=True,
            )
            result_list.append(
                f'visual session:{session}, epoch:{epoch}, train_loss:{train_loss:.5f}, '
                f'train_acc:{train_acc:.5f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.5f}'
            )
            self.result_log = result_list
            self.log_event(
                'visual',
                session,
                epoch=int(epoch),
                train_loss=round(train_loss, 5),
                train_acc=round(train_acc, 5),
                test_loss=round(test_loss, 5),
                test_acc=round(test_acc, 5),
            )
            if test_acc > best_acc:
                best_acc = test_acc
                self.visual_max_acc[session] = float(f'{test_acc * 100:.3f}')
                best_state = deepcopy(self.model.state_dict())
                self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_visual_best.pth'))

        self.model.load_state_dict(best_state)
        return best_acc

    def run_router_stage(self, session, trainloader, testloader, result_list):
        route_prototypes = self.collect_route_prototypes(trainloader, session)
        self.update_router_memory(trainloader, session, prototypes=route_prototypes)

        if session == 0 and self.args.router_disc_type == 'msd':
            test_loss, test_acc = eval_router(self.model, testloader, self.args, session)
            self.router_max_acc[session] = float(f'{test_acc * 100:.3f}')
            print(
                f'[router] session={session} skipped train stage for single-session MSD, '
                f'test_loss={test_loss:.5f} test_acc={test_acc:.5f}',
                flush=True,
            )
            result_list.append(
                f'router session:{session}, epoch:skip, train_loss:0.00000, train_acc:1.00000, '
                f'test_loss:{test_loss:.5f}, test_acc:{test_acc:.5f}'
            )
            self.result_log = result_list
            self.log_event(
                'router',
                session,
                epoch='skip',
                train_loss=0.0,
                train_acc=1.0,
                test_loss=round(test_loss, 5),
                test_acc=round(test_acc, 5),
            )
            return test_acc

        if self.args.router_feat_mode == 'warp' and hasattr(self.model.module.router.feature_extractor, 'compute_basis'):
            self.model.module.router.feature_extractor.compute_basis(
                trainloader,
                max_batches=self.args.warp_basis_max_batches,
            )

        optimizer, scheduler = self.get_router_optimizer(session)
        epochs = self._epoch_count(session, 'router')
        best_acc = -1.0
        best_state = deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            train_loss, train_acc = run_router_epoch(
                self.model,
                trainloader,
                optimizer,
                scheduler,
                epoch,
                self.args,
                session,
                train=True,
                route_prototypes=route_prototypes,
                scaler=self.scaler,
            )
            route_prototypes = self.collect_route_prototypes(trainloader, session)
            self.update_router_memory(trainloader, session, prototypes=route_prototypes)
            test_loss, test_acc = eval_router(self.model, testloader, self.args, session)
            print(
                f'[router] session={session} epoch={epoch} '
                f'train_loss={train_loss:.5f} train_acc={train_acc:.5f} '
                f'test_loss={test_loss:.5f} test_acc={test_acc:.5f}',
                flush=True,
            )
            result_list.append(
                f'router session:{session}, epoch:{epoch}, train_loss:{train_loss:.5f}, '
                f'train_acc:{train_acc:.5f}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.5f}'
            )
            self.result_log = result_list
            self.log_event(
                'router',
                session,
                epoch=int(epoch),
                train_loss=round(train_loss, 5),
                train_acc=round(train_acc, 5),
                test_loss=round(test_loss, 5),
                test_acc=round(test_acc, 5),
            )
            if test_acc > best_acc:
                best_acc = test_acc
                self.router_max_acc[session] = float(f'{test_acc * 100:.3f}')
                best_state = deepcopy(self.model.state_dict())
                self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_router_best.pth'))

        self.model.load_state_dict(best_state)
        route_prototypes = self.collect_route_prototypes(trainloader, session)
        self.update_router_memory(trainloader, session, prototypes=route_prototypes)
        if self.args.router_feat_mode == 'warp':
            keep_ratio = self.update_router_warp_mask(trainloader, session, route_prototypes)
            if keep_ratio is not None:
                print(
                    f'[router] session={session} warp_keep_ratio={keep_ratio:.5f} '
                    f'(target={self.args.fraction_to_keep:.5f})',
                    flush=True,
                )
                result_list.append(
                    f'router session:{session}, warp_keep_ratio:{keep_ratio:.5f}, '
                    f'warp_target_keep_ratio:{self.args.fraction_to_keep:.5f}'
                )
                self.result_log = result_list
                self.log_event(
                    'router_warp',
                    session,
                    warp_keep_ratio=round(keep_ratio, 5),
                    warp_target_keep_ratio=round(self.args.fraction_to_keep, 5),
                )
            if hasattr(self.model.module.router.feature_extractor, 'restore_weights'):
                self.model.module.router.feature_extractor.restore_weights()
        return best_acc

    def evaluate_session_overall(self, session, testloader, result_list):
        current_state = deepcopy(self.model.state_dict())
        current_extra_state = self._extra_state()
        checkpoint_dir = Path(self.args.save_path)
        candidate_paths = [
            ('visual_best', checkpoint_dir / f'session{session}_visual_best.pth'),
            ('router_best', checkpoint_dir / f'session{session}_router_best.pth'),
        ]

        best_name = None
        best_loss = None
        best_acc = -1.0

        for name, path in candidate_paths:
            if not path.is_file():
                continue
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['params'] if isinstance(checkpoint, dict) and 'params' in checkpoint else checkpoint
            extra_state = checkpoint.get('extra_state', {}) if isinstance(checkpoint, dict) else {}
            self.model.load_state_dict(state_dict, strict=False)
            self._load_extra_state(extra_state)
            self.model.module.active_session = session
            test_loss, test_acc = eval_overall(self.model, testloader, self.args, session)
            print(
                f'[overall] session={session} source={name} test_loss={test_loss:.5f} test_acc={test_acc:.5f}',
                flush=True,
            )
            result_list.append(
                f'overall session:{session}, source:{name}, test_loss:{test_loss:.5f}, test_acc:{test_acc:.5f}'
            )
            self.result_log = result_list
            self.log_event(
                'overall_candidate',
                session,
                source=name,
                test_loss=round(test_loss, 5),
                test_acc=round(test_acc, 5),
            )
            if test_acc > best_acc:
                best_name = name
                best_loss = test_loss
                best_acc = test_acc
                self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_overall_best.pth'))

        self.model.load_state_dict(current_state, strict=False)
        self._load_extra_state(current_extra_state)

        if best_name is None:
            self.model.module.active_session = session
            return None

        overall_path = os.path.join(self.args.save_path, f'session{session}_overall_best.pth')
        checkpoint = torch.load(overall_path, map_location='cpu')
        state_dict = checkpoint['params'] if isinstance(checkpoint, dict) and 'params' in checkpoint else checkpoint
        extra_state = checkpoint.get('extra_state', {}) if isinstance(checkpoint, dict) else {}
        self.model.load_state_dict(state_dict, strict=False)
        self._load_extra_state(extra_state)
        self.model.module.active_session = session
        self.overall_max_acc[session] = float(f'{best_acc * 100:.3f}')
        print(
            f'[overall] session={session} selected={best_name} test_loss={best_loss:.5f} test_acc={best_acc:.5f}',
            flush=True,
        )
        result_list.append(
            f'overall session:{session}, selected:{best_name}, test_loss:{best_loss:.5f}, test_acc:{best_acc:.5f}'
        )
        self.result_log = result_list
        self.log_event(
            'overall_selected',
            session,
            source=best_name,
            test_loss=round(best_loss, 5),
            test_acc=round(best_acc, 5),
        )
        return best_acc

    def train(self):
        result_list = list(self.result_log)
        for session in range(self.args.start_session, self.args.sessions):
            self.model.module.active_session = session
            _, trainloader, testloader = self.get_dataloader(session)
            self.run_visual_stage(session, trainloader, testloader, result_list)
            self.run_router_stage(session, trainloader, testloader, result_list)
            self.evaluate_session_overall(session, testloader, result_list)
            self.best_model_dict = deepcopy(self.model.state_dict())
            self.log_event('session_end', session, checkpoint=f'session{session}_last.pth')
            self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_last.pth'))

        result_list.append({
            'visual_max_acc': self.visual_max_acc,
            'router_max_acc': self.router_max_acc,
            'overall_max_acc': self.overall_max_acc,
        })
        self.result_log = result_list
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)
        self.persist_training_log(snapshot_only=True)
