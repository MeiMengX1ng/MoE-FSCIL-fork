# MoE-FSCIL

**基于混合专家模型的小样本类增量学习算法**
*(Few-Shot Class-Incremental Learning based on Mixture-of-Experts)*

本仓库核心贡献在于将视觉 Transformer 骨干、阶段专属低秩专家（LoRA-EM）、阶段路由模块（SRM）、阶段解耦分类模块（SDC）以及原型近邻插值增广算法（PNN-FA）整合于同一框架，并采用 **"视觉阶段 → 路由阶段"两阶段交替训练**范式，以系统性地缓解 FSCIL 任务中的灾难性遗忘与新类偏置问题。

---

## 1. 方法概览

### 1.1 问题定义

给定按阶段（session）`S = {0, 1, …, T}` 顺序到达的数据流，基类阶段 `t = 0` 提供充足的标注数据，增量阶段 `t ≥ 1` 仅提供少量新类样本（典型 N-way K-shot，本仓库默认 `K = 5`）。模型在每个阶段 `t` 需同时识别已学类别 `C^(0:t)` 中所有类，且无法回放旧类原始数据。

### 1.2 整体框架

本方法遵循"**显式参数隔离 + 隐式特征增广**"的双层范式：

| 模块 | 位置 | 功能 | 缓解问题 |
| --- | --- | --- | --- |
| 低秩专家模块（LoRA-EM） | ViT FFN 内部 | 为每个阶段分配专属的 `ΔW^(t) = B^(t) A^(t)` 旁路 | 参数级隔离新/旧知识 |
| 阶段路由模块（SRM） | 骨干外部 | 推断输入样本所属阶段，激活对应专家 | 推理期阶段标识缺失 |
| 阶段解耦分类模块（SDC） | 分类头 | 显式拆分阶段子分类器 `C^(t)(·)`，施加几何正交约束 | 决策边界向旧类坍缩 |
| 原型近邻插值增广（PNN-FA） | MSD 训练期 | 在 K 近邻类原型方向上线性插值 | 小样本特征稀疏 |
| WaRP 正交基重构 | 路由特征提取 | 将预训练卷积权重在输入协方差正交基下重新参数化 | 缓解路由特征随训练漂移 |

整体推理管线：

```text
x ─┬─► [Frozen ResNet-18 (±WaRP)] ──► SRM.Discriminator ──► t*        (路由分支)
   └─► [CLIP ViT-B/16 + LoRA^(t*)] ──► f ──► C^(t*)(f) ──► ŷ              (分类分支)
```

### 1.3 关键公式

低秩专家旁路：

```
y = W₀·x + α · B^(t)(A^(t)·x),    r ≪ min(d_out, d_in)
```

阶段间正交约束：

```
ℒ_ortho = ‖Wₙ·Wₙᵀ − I‖² + λ_cross · ‖Wₙ·Wₒᵀ‖²
```

原型近邻插值：

```
f̃ = f + λ · Δf,    Δf = ½(c − c*),    λ ~ U[λ_min, λ_max]
```

DSD 阶段判别：

```
t* = arg min_{t ∈ {0, …, T}}  ‖f_in − ℱ_Aᵗ(f_in)‖²
```

---

## 2. 代码结构

| 文件 | 角色 | 主要类/函数 |
| --- | --- | --- |
| [models/vmoe/Network.py](models/vmoe/Network.py) | 顶层网络 | `FrozenVisionBackbone`, `VMOENet`（含 `_forward_mixed_sessions`） |
| [models/vmoe/clip_loader.py](models/vmoe/clip_loader.py) | CLIP 视觉骨干加载 | `load_clip_vision_backbone`（含 `HF_FALLBACK_CACHE` 离线快照与 `CLIP_VIT_B16_PATH` 覆盖） |
| [models/vmoe/lora.py](models/vmoe/lora.py) | 低秩专家实现 | `SessionLoRAExpert`, `MultiSessionLoRA`, `LoRALinear`, `inject_lora_to_vit_encoder_layers` |
| [models/vmoe/router.py](models/vmoe/router.py) | 阶段路由 | `ResNet18FeatureExtractor`, `FrozenResNet18Extractor`, `WaRPResNet18Extractor`, `MemoryBankSessionDiscriminator`, `DistributionSessionDiscriminator`, `SessionAutoEncoder`, `SessionRoutingModule` |
| [models/vmoe/classifier.py](models/vmoe/classifier.py) | 阶段解耦分类 | `SessionDecoupledClassifier`（含 `orthogonality_loss`） |
| [models/vmoe/augment.py](models/vmoe/augment.py) | 原型增广 | `build_class_prototypes`, `prototype_nearest_neighbor_augment` |
| [models/vmoe/warp.py](models/vmoe/warp.py) | WaRP 路由特征提取 | `WaRPModule`, `Conv2dWaRP`, `compute_warp_orthonormal_basis`, `restore_warp_weights`, `switch_warp_modules` |
| [models/vmoe/fscil_trainer.py](models/vmoe/fscil_trainer.py) | FSCIL 训练循环 | `FSCILTrainer`（含 `run_visual_stage` / `run_router_stage` 两阶段调度） |
| [models/vmoe/helper.py](models/vmoe/helper.py) | 训练/评估 epoch | `run_visual_epoch`, `eval_visual`, `run_router_epoch`, `eval_router` |
| [dataloader/data_utils.py](dataloader/data_utils.py) | 数据集注册与 DataLoader | `set_up_datasets`, `get_base_dataloader`, `get_new_dataloader` |
| [dataloader/cifar100/cifar.py](dataloader/cifar100/cifar.py) | CIFAR-100 数据集 | `CIFAR100`（含 `NewClassSelector`） |
| [train.py](train.py) | 命令行入口 | `get_command_line_parser` |

### 2.1 模块对应关系

| 模块 | 实现位置 |
| --- | --- |
| 低秩专家模块 | [lora.py:8-117](models/vmoe/lora.py#L8-L117) |
| 阶段路由模块 | [router.py:137-170](models/vmoe/router.py#L137-L170) |
| 阶段解耦分类模块 | [classifier.py:6-57](models/vmoe/classifier.py#L6-L57) |
| 基于记忆存储器的阶段判别器 (MSD) | [router.py:67-95](models/vmoe/router.py#L67-L95) |
| 基于分布判别的阶段判别器 (DSD) | [router.py:98-134](models/vmoe/router.py#L98-L134) |
| 原型近邻插值特征增广 (PNN-FA) | [augment.py:14-45](models/vmoe/augment.py#L14-L45) |
| WaRP 路由特征提取 | [warp.py:31-103](models/vmoe/warp.py#L31-L103) |
| CLIP 视觉骨干加载 | [clip_loader.py:69-80](models/vmoe/clip_loader.py#L69-L80) |
| 两阶段训练调度 | [fscil_trainer.py:211-312](models/vmoe/fscil_trainer.py#L211-L312) |

---

## 3. 复现实验

### 3.1 硬件与软件环境

| 项目 | 配置 |
| --- | --- |
| CPU | Intel(R) Core(TM) i9-10920X |
| 内存 | 256 GB |
| GPU | 4 × NVIDIA RTX 2080 Ti |
| 框架 | PyTorch + `transformers`（HuggingFace） |
| 关键依赖 | `openai/clip-vit-base-patch16`, `torchvision` ResNet-18 |

### 3.2 数据集

| 数据集 | 基类数 | 增量配置 | 总阶段数（含基类） | shot | 评估集 |
| --- | --- | --- | --- | --- | --- |
| CIFAR-100 | 60 | 5-way × 8 增量 | 9 | 5 | 全部已见类 |
| MiniImageNet | 60 | 5-way × 8 增量 | 9 | 5 | 全部已见类 |
| CUB-200 | 100 | 10-way × 10 增量 | 11 | 5 | 全部已见类 |

所有图像最终输入到 CLIP 骨干时统一插值到 224×224；归一化因数据集而异（CIFAR-100 使用 CLIP 默认 mean/std，其余使用 ImageNet mean/std），具体如下：

| 数据集 | 训练期增广 | 测试期 |
| --- | --- | --- |
| CIFAR-100 | `Resize(224) → RandomCrop(224, padding=28) → RandomHorizontalFlip → Normalize(CLIP)` | `Resize(224) → Normalize(CLIP)` |
| MiniImageNet | `RandomResizedCrop(image_size) → RandomHorizontalFlip → Normalize(ImageNet)` | `Resize(92) → CenterCrop(image_size) → Normalize(ImageNet)` |
| CUB-200 | `Resize(256) → RandomResizedCrop(224) → RandomHorizontalFlip → Normalize(ImageNet)` | `Resize(256) → CenterCrop(224) → Normalize(ImageNet)` |

新类样本通过 `data/index_list/<dataset>/session_{t+1}.txt` 中按 `(way, shot)` 形状排列的索引切片加载；CIFAR-100 使用 `NewClassSelector`（5×5=25 索引）直接从 `cifar-100-python` 切分，其余数据集走 `index_path`。

### 3.3 训练超参（视觉阶段与路由阶段共享骨架）

| 参数 | MiniImageNet | CIFAR-100 | CUB-200 |
| --- | --- | --- | --- |
| 优化器 | SGD (momentum 0.9, weight decay 5×10⁻⁴) | 同左 | 同左 |
| 学习率调度 | `Milestone`（`MultiStepLR`，γ=0.1） | 同左 | 同左 |
| 学习率衰减节点（基类） | [40, 80] | [40, 80] | [30, 60, 90] |
| 学习率衰减节点（新类） | [10, 15] | [10, 15] | [10, 15] |
| 初始学习率（基类 / 新类） | 0.1 / 0.1 | 0.01 / 0.01 | 0.01 / 0.01 |
| 基类训练轮数 | 100 | 100 | 120 |
| 新类训练轮数 | 20 | 20 | 20 |
| 批大小（基类） | 128 | 256 | 128 |
| 批大小（新类） | 0 = 全量 | 0 = 全量 | 0 = 全量 |
| 测试批大小 | 100 | 256 | 100 |
| `base_mode` / `new_mode` | `ft_cos` / `ft_cos` | 同左 | 同左 |
| 余弦温度 τ | 16 | 16 | 16 |
| AMP（混合精度） | 启用 | 启用 | 启用 |

### 3.4 路由与增广超参

| 参数 | MSD | DSD |
| --- | --- | --- |
| 路由特征提取 | `frozen`（ImageNet 预训练 ResNet-18）或 `warp` | 同左 |
| 路由优化器 | SGD | SGD（与视觉阶段共享 `lr_*` / `milestones_new`） |
| 路由训练轮数（基类） | 30（CUB：40） | 30（CUB：40） |
| 路由训练轮数（新类） | 10 | 100（CIFAR-100）/ 10（其余） |
| 损失函数 | 交叉熵（带 PNN-FA） | 交叉熵（基于重构误差） |
| 路由特征维度 | 512 | 512 |
| 瓶颈维度 | — | 256 |
| K（近邻数） | 5 | 10 |
| N（采样次数） | 5 | 5 |
| λ 范围 | [0.45, 0.75] | [0.45, 0.75] |
| 基类特殊处理 | `session=0` 跳过训练（仅评估） | 正常训练 AE₀ |
| `num_workers` | 16 | 16（CIFAR-100 启用 100） |

正交损失系数：基类 `λ_ortho_base = 0.01`，新类 `λ_ortho_new = 0.05`，跨阶段 `λ_cross = 0.8`。

### 3.5 两阶段训练流程

`FSCILTrainer.train()` 按以下顺序对 `t = start_session … sessions-1` 逐阶段执行：

1. **构建 DataLoader**：`session=0` 调用 `get_base_dataloader`，其余调用 `get_new_dataloader`；新类批大小为 0 时一次性送入全部样本。
2. **视觉阶段 `run_visual_stage`**（[fscil_trainer.py:211-247](models/vmoe/fscil_trainer.py#L211-L247)）：
   - 优化对象：当前阶段 `t` 的 `SessionLoRAExpert.A/B` 参数 + `SessionDecoupledClassifier.heads[t]`。
   - 损失：`CE(logits[:, :seen_classes], label) + λ_ortho · ℒ_ortho(t)`，正交损失同时含类内自正交与跨阶段互正交。
   - 调度：SGD + `MultiStepLR(milestones, gamma=0.1)`，共 `epochs_base / epochs_new` 轮。
   - 监控：在测试集上做 `eval_visual`，保留 `test_acc` 最高的 `session{t}_visual_best.pth`。
3. **路由阶段 `run_router_stage`**（[fscil_trainer.py:249-312](models/vmoe/fscil_trainer.py#L249-L312)）：
   - 先用视觉阶段训完后的模型前向收集 `route_prototypes = build_class_prototypes(pooled_features, labels, class_ids)`。
   - 将 `prototypes` 写入 MSD 的 `memory_bank[t]`，覆盖式中以 `session_memory_overrides={t: prototypes}` 提供。
   - 若 `-router_feat_mode warp`，调用 `WaRPResNet18Extractor.compute_basis(trainloader)` 在输入协方差正交基下重参数化卷积层。
   - **MSD 基类短路**：`session=0` 时只有一个类簇，路由训练无意义，代码直接 `eval_router` 后跳过优化（[fscil_trainer.py:256-268](models/vmoe/fscil_trainer.py#L256-L268)）；DSD 基类仍正常训练 AE₀。
   - 优化对象：ResNet-18 路由特征提取器（解冻全部 BN/Conv）+ 判别器参数（MSD 的 `projector` / DSD 的 `autoencoders[t]`）。
   - 损失：MSD 为 `CE(logits, labels)` 且仅在 `session>0` 时启用 PNN-FA；DSD 为 `CE(-MSE_recon_error, session_targets)`。
   - 调度：与视觉阶段共享 `lr_*` / `milestones_new`。
   - 每个 epoch 后重新收集 `route_prototypes` 并更新 `memory_bank`，保留 `session{t}_router_best.pth`。
   - 路由阶段结束：若使用 `warp`，调用 `restore_warp_weights` 恢复原 Conv 权重，避免污染下一阶段。
4. **保存 `session{t}_last.pth`**：包含 `params`（视觉+路由全部参数）和 `extra_state`（`active_session`、MSD 的 `memory_bank`），用于断点续训。
5. **最终记录**：`results.txt` 同时写入 `visual_max_acc` 与 `router_max_acc` 两组逐阶段最佳值。

### 3.6 启动方式

```bash
# 三个数据集的官方脚本入口
bash script/cifar100.sh
bash script/mini_imagenet.sh
bash script/cub200.sh
```

脚本顶部可覆盖的关键开关：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `router_disc_type` | `msd` | `msd` / `dsd` |
| `router_feat_mode` | `frozen` | `frozen` / `warp` |
| `router_neighbor_k` | 5 (msd) / 10 (dsd) | PNN-FA 近邻数 K |
| `router_sample_n` | 5 | PNN-FA 采样次数 N |
| `router_epochs_base` | 30（`cub200.sh`: 40） | 路由基类训练轮数 |
| `router_epochs_new` | 10（`cifar100.sh` dsd: 100） | 路由新类训练轮数 |
| `num_workers` | 16（`cifar100.sh` dsd: 100） | DataLoader worker 数 |
| `model_dir` | 空 | 设为路径则断点续训（自动选 `session{t-1}_*.pth`） |
| `gpu_num` | 0 / 2 | 可见 GPU 列表，传给 `-gpu` |

也可手动指定：

```bash
python train.py \
    -project vmoe \
    -dataset {cifar100|mini_imagenet|cub200} \
    -backbone_type clip_vit_b16 \
    -model_image_size 224 \
    -router_disc_type {msd|dsd} \
    -router_feat_mode {frozen|warp} \
    -lora_rank 8 -lora_alpha 16 \
    -lambda_cross 0.8 \
    -lambda_ortho_base 0.01 -lambda_ortho_new 0.05 \
    -epochs_base 100 -epochs_new 20 \
    -lr_base 0.01 -lr_new 0.01 \
    -batch_size_base 128 -test_batch_size 100 \
    -temperature 16 -gpu 0
```

### 3.7 推理流程

1. 阶段路由：图像经冻结 `ResNet-18`（或 `WaRP ResNet-18`）提取 512 维视觉特征（`AdaptiveAvgPool2d → flatten`），由 `MemoryBankSessionDiscriminator` 或 `DistributionSessionDiscriminator.predict_session` 推断阶段 `t*`。
2. 专家激活：在 CLIP ViT-B/16 的所有 FFN 中调用 `LoRALinear.set_active_session(t*)`，仅 `t*` 对应的 `SessionLoRAExpert` 参与前向，其余 `ΔW = 0`。
3. 类别判别：768 维 `[CLS]` token 经 `SessionDecoupledClassifier` 将 `heads[0..t*]` 的权重按 `base_class / way` 切片拼接，与归一化特征做余弦相似度（×温度 τ）后取 `arg max`。
4. 混合阶段 batch：`VMOENet._forward_mixed_sessions` 按 `session_ids` 分组，分别激活对应 LoRA 与子分类头后再拼接 logits（[Network.py:93-103](models/vmoe/Network.py#L93-L103)）。
5. 检查点恢复：`FSCILTrainer.__init__` 通过 `resolve_checkpoint_path` 自动从 `-model_dir` 选最近一个 `session*_last.pth`（或显式 `session*_visual_best.pth` / `session*_router_best.pth`），并把 `extra_state.memory_bank` 写回 MSD 判别器，保证断点续训时路由一致。

---

## 4. 关键超参说明

| CLI 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-backbone_type` | `clip_vit_b16` | 当前仅支持 CLIP ViT-B/16 视觉骨干 |
| `-model_image_size` | 224 | CLIP 输入尺寸（与 `CLIP_IMAGE_SIZE` 一致） |
| `-base_mode` / `-new_mode` | `ft_cos` / `ft_cos` | 头部模式：`ft_cos`（带温度的余弦）/ `ft_dot`（内积）/`avg_cos`（原型平均） |
| `-epochs_base` | 100 | 基类视觉阶段轮数 |
| `-epochs_new` | 20 | 新类视觉阶段轮数 |
| `-lr_base` | 0.1 | 视觉阶段基类学习率（脚本：CIFAR/CUB 改 0.01） |
| `-lr_new` | 0.1 | 视觉阶段新类学习率（脚本：CIFAR/CUB 改 0.01） |
| `-momentum` | 0.9 | SGD 动量 |
| `-decay` | 5×10⁻⁴ | SGD 权重衰减 |
| `-gamma` | 0.1 | `MultiStepLR` 学习率衰减系数 |
| `-schedule` | `Milestone` | 当前仅启用 `MultiStepLR` 分支（`Step` 保留兼容） |
| `-milestones` | `[60, 70]` | 视觉阶段基类 LR 衰减节点（脚本：`[40, 80]` / CUB `[30, 60, 90]`） |
| `-milestones_new` | `[10, 15]` | 视觉阶段新类与路由阶段共享的 LR 衰减节点 |
| `-batch_size_base` | 128 | 基类训练批大小（脚本：CIFAR-100 改 256） |
| `-batch_size_new` | 0 | 新类批大小；0 = 全量一次性送入 |
| `-test_batch_size` | 100 | 评估批大小（脚本：CIFAR-100 改 256） |
| `-temperature` | 16 | 分类余弦温度 τ |
| `-lora_rank` | 8 | 低秩分解的秩 r |
| `-lora_alpha` | 8.0（脚本覆盖为 16） | 专家输出缩放系数 α（实际缩放为 α/r） |
| `-lambda_cross` | 0.8 | 跨阶段正交系数 λ_cross |
| `-lambda_ortho_base` | 0.01 | 基类阶段正交损失权重 |
| `-lambda_ortho_new` | 0.05 | 增量阶段正交损失权重 |
| `-aug_lambda_min` | 0.45 | PNN-FA 插值强度下界 |
| `-aug_lambda_max` | 0.75 | PNN-FA 插值强度上界 |
| `-router_disc_type` | `msd` | `msd` = 记忆存储器判别器；`dsd` = 分布差异判别器 |
| `-router_feat_mode` | `frozen` | `frozen` = 冻结 ResNet-18；`warp` = 在输入协方差正交基下重参数化 |
| `-router_bottleneck_dim` | 256 | DSD 中自动编码器的瓶颈维度 |
| `-router_loss_weight` | 1.0 | 路由损失在总损失中的权重（保留位，当前未拼接联合损失） |
| `-router_neighbor_k` | 5（脚本：DSD 改 10） | PNN-FA 原型近邻数 K |
| `-router_sample_n` | 5 | 每个样本生成的偏移样本数 N |
| `-router_epochs_base` | 30（脚本：CUB 改 40） | 路由阶段基类轮数 |
| `-router_epochs_new` | 10（脚本：CIFAR-100 DSD 改 100） | 路由阶段新类轮数 |
| `-start_session` | 0 | 起始阶段；非 0 时配合 `-model_dir` 实现断点续训 |
| `-model_dir` | `None` | 检查点路径；为空则随机初始化并加载 CLIP/ResNet-18 预训练权重 |
| `-gpu` | `0,1,2,3` | 可见 GPU 列表，多卡走 `nn.DataParallel` |
| `-num_workers` | 16 | DataLoader worker 数（脚本：CIFAR-100 DSD 改 100） |
| `-seed` | 1 | 随机种子 |
| `-debug` | False | 调试开关 |

---

## 5. 许可

本仓库仅供学术研究使用，CLIP（OpenAI）与 ResNet-18（torchvision）的预训练权重遵循其各自原始许可，CIFAR-100/MiniImageNet/CUB-200 数据集遵循其原始数据集协议。
