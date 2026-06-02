import os
from copy import deepcopy

import torch
import torch.nn as nn

from dataloader.data_utils import get_base_dataloader, get_new_dataloader, set_up_datasets
from models.base.base import Trainer
from utils import ensure_path, save_list_to_txt
from .Network import VMOENet
from .augment import build_class_prototypes
from .helper import eval_model, run_epoch


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = set_up_datasets(args)
        self.set_save_path()
        self.model = nn.DataParallel(VMOENet(self.args, mode=self.args.base_mode), list(range(self.args.num_gpu))).cuda()
        if self.args.model_dir is not None:
            checkpoint = torch.load(self.args.model_dir, map_location='cpu')
            self.best_model_dict = checkpoint['params']
            self.model.load_state_dict(self.best_model_dict, strict=False)
            self._load_extra_state(checkpoint.get('extra_state', {}))
        else:
            self.best_model_dict = deepcopy(self.model.state_dict())

    def set_save_path(self):
        checkpoint_root = '/data/xuzg/FSCIL/MoE-FSCIL/checkpoint'
        router_tag = f"{self.args.router_disc_type}-routerpt"
        lora_tag = f"lora_r{self.args.lora_rank}_a{self.args.lora_alpha:.1f}"
        ortho_tag = (
            f"ortho_b{self.args.lambda_ortho_base:g}_"
            f"n{self.args.lambda_ortho_new:g}_"
            f"cross{self.args.lambda_cross:g}"
        )
        mode_tag = 'moe-moe-data_init-start_0'
        run_name = f"{router_tag}-{lora_tag}-{ortho_tag}-{mode_tag}"
        self.args.save_path = os.path.join(checkpoint_root, self.args.dataset, self.args.project, run_name)
        ensure_path(self.args.save_path)

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
            },
            path,
        )

    def get_dataloader(self, session):
        if session == 0:
            return get_base_dataloader(self.args)
        return get_new_dataloader(self.args, session)

    def get_optimizer(self, session):
        params = []
        if self.model.module.use_backbone_lora:
            params.extend(self.model.module.backbone.trainable_parameters_for_session(session))
        else:
            params.extend(self.model.module.lora.trainable_parameters_for_session(session))
        params.extend(self.model.module.classifier.heads[session].parameters())
        if self.args.router_feat_mode == 'warp':
            params.extend(self.model.module.router.feature_extractor.parameters())

        if self.args.router_disc_type == 'msd':
            params.extend(self.model.module.router.discriminator.projector.parameters())
        else:
            params.extend(self.model.module.router.discriminator.autoencoders[session].parameters())

        optimizer = torch.optim.SGD(
            params,
            lr=self.args.lr_base if session == 0 else self.args.lr_new,
            momentum=self.args.momentum,
            weight_decay=self.args.decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.args.milestones if session == 0 else self.args.milestones_new,
            gamma=self.args.gamma,
        )
        return optimizer, scheduler

    def update_router_memory(self, trainloader, session):
        if self.args.router_disc_type != 'msd':
            return

        feats = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in trainloader:
                images, target = [_.cuda() for _ in batch]
                _, pooled = self.model.module.router.extract_route_features(images)
                feats.append(pooled)
                labels.append(target)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        prototypes = build_class_prototypes(feats, labels, self.model.module.get_session_class_ids(session))
        self.model.module.router.discriminator.update_memory(session, prototypes)

    def train(self):
        result_list = [self.args]
        for session in range(self.args.start_session, self.args.sessions):
            self.model.module.active_session = session
            train_set, trainloader, testloader = self.get_dataloader(session)
            if self.args.router_feat_mode == 'warp' and hasattr(self.model.module.router.feature_extractor, 'compute_basis'):
                self.model.module.router.feature_extractor.compute_basis(trainloader)
            optimizer, scheduler = self.get_optimizer(session)
            epochs = self.args.epochs_base if session == 0 else self.args.epochs_new

            for epoch in range(epochs):
                tl, ta = run_epoch(self.model, trainloader, optimizer, scheduler, epoch, self.args, session, train=True)
                vl, va = eval_model(self.model, testloader, self.args, session)
                result_list.append(
                    f"session:{session}, epoch:{epoch}, train_loss:{tl:.5f}, train_acc:{ta:.5f}, test_loss:{vl:.5f}, test_acc:{va:.5f}"
                )
                if va * 100 >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float(f"{va * 100:.3f}")
                    self.trlog['max_acc_epoch'] = epoch
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_max_acc.pth'))

            self.model.load_state_dict(self.best_model_dict)
            self.update_router_memory(trainloader, session)
            if self.args.router_feat_mode == 'warp' and hasattr(self.model.module.router.feature_extractor, 'restore_weights'):
                self.model.module.router.feature_extractor.restore_weights()
            self.best_model_dict = deepcopy(self.model.state_dict())
            self.save_checkpoint(os.path.join(self.args.save_path, f'session{session}_last.pth'))

        result_list.append(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)
