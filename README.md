# MoE-FSCIL

**基于混合专家模型的小样本类增量学习算法**
*(Few-Shot Class-Incremental Learning based on Mixture-of-Experts)*

本仓库核心贡献在于将视觉 Transformer 骨干、阶段专属低秩专家（LoRA-EM）、阶段路由模块（SRM）、阶段解耦分类模块（SDC）以及原型近邻插值增广算法（PNN-FA）整合于同一框架，以系统性地缓解 FSCIL 任务中的灾难性遗忘与新类偏置问题。

---

## 1. 方法概览

### 1.1 问题定义

给定按阶段（session）$\mathcal{S}=\{0,1,\dots,T\}$ 顺序到达的数据流，基类阶段 $t=0$ 提供充足的标注数据，增量阶段 $t\ge 1$ 仅提供少量新类样本（典型 $N$-way $K$-shot，$K=1$）。模型在每个阶段 $t$ 需同时识别已学类别 $\mathcal{C}^{(0:t)}$ 中所有类，且无法回放旧类原始数据。

### 1.2 整体框架

本方法遵循"**显式参数隔离 + 隐式特征增广**"的双层范式：

| 模块 | 位置 | 功能 | 缓解问题 |
| --- | --- | --- | --- |
| 低秩专家模块（LoRA-EM） | ViT FFN 内部 | 为每个阶段分配专属的 $\Delta W^{(t)}=B^{(t)}A^{(t)}$ 旁路 | 参数级隔离新/旧知识 |
| 阶段路由模块（SRM） | 骨干外部 | 推断输入样本所属阶段，激活对应专家 | 推理期阶段标识缺失 |
| 阶段解耦分类模块（SDC） | 分类头 | 显式拆分阶段子分类器 $C^{(t)}(\cdot)$，施加几何正交约束 | 决策边界向旧类坍缩 |
| 原型近邻插值增广（PNN-FA） | 训练期 | 在 $K$ 近邻类原型方向上线性插值 | 小样本特征稀疏 |

整体推理管线：

```text
x ─┬─► [Frozen ResNet-18] ──► SRM.Discriminator ──► t*        (路由分支)
   └─► [CLIP ViT-B/16 + LoRA^{(t*)}] ──► f ──► C^{(t*)}(f) ──► ŷ   (分类分支)
```

### 1.3 关键公式

低秩专家旁路：

$$y = W_0 x + \alpha \cdot B^{(t)}\big(A^{(t)} x\big),\quad r \ll \min(d_\text{out}, d_\text{in})$$

阶段间正交约束（公式 3-6）：

$$\mathcal{L}_\text{ortho} = \big\|W_n W_n^\top - I\big\|_2^2 + \lambda_\text{cross}\big\|W_n W_o^\top\big\|_2^2$$

原型近邻插值（公式 3-12、3-13）：

$$\widetilde{f} = f + \lambda\,\Delta f,\qquad \Delta f = \tfrac{1}{2}\big(c - c^*\big),\quad \lambda \sim \mathcal{U}[\lambda_\text{min},\lambda_\text{max}]$$

DSD 阶段判别（公式 3-11）：

$$t^* = \arg\min_{t\in\{0,\dots,T\}} \big\|f_\text{in} - \mathcal{F}_A^{t}(f_\text{in})\big\|_2^2$$

---

## 2. 代码结构

| 文件 | 角色 | 主要类/函数 |
| --- | --- | --- |
| [models/vmoe/Network.py](models/vmoe/Network.py) | 顶层网络 | `FrozenVisionBackbone`, `VMOENet` |
| [models/vmoe/lora.py](models/vmoe/lora.py) | 低秩专家实现 | `SessionLoRAExpert`, `MultiSessionLoRA`, `LoRALinear`, `inject_lora_to_vit_encoder_layers` |
| [models/vmoe/router.py](models/vmoe/router.py) | 阶段路由 | `FrozenResNet18Extractor`, `WaRPResNet18Extractor`, `MemoryBankSessionDiscriminator`, `DistributionSessionDiscriminator`, `SessionRoutingModule` |
| [models/vmoe/classifier.py](models/vmoe/classifier.py) | 阶段解耦分类 | `SessionDecoupledClassifier`（含 `orthogonality_loss`） |
| [models/vmoe/augment.py](models/vmoe/augment.py) | 原型增广 | `build_class_prototypes`, `prototype_nearest_neighbor_augment` |
| [models/vmoe/warp.py](models/vmoe/warp.py) | WaRP 路由特征提取 | `WaRPModule`, `Conv2dWaRP`, `compute_warp_orthonormal_basis` |
| [models/vmoe/fscil_trainer.py](models/vmoe/fscil_trainer.py) | FSCIL 训练循环 | `FSCILTrainer` |
| [models/vmoe/helper.py](models/vmoe/helper.py) | 训练/评估 epoch | `run_epoch`, `eval_model` |
| [train.py](train.py) | 命令行入口 | `get_command_line_parser` |

### 2.1 模块对应关系

| 论文小节 | 实现位置 |
| --- | --- |
| 3.2.2 低秩专家模块 | [lora.py:8-117](models/vmoe/lora.py#L8-L117) |
| 3.2.3 阶段路由模块 | [router.py:9-169](models/vmoe/router.py#L9-L169) |
| 3.2.4 阶段解耦分类模块 | [classifier.py:6-57](models/vmoe/classifier.py#L6-L57) |
| 3.3.1 基于记忆存储器的阶段判别器 (MSD) | [router.py:67-94](models/vmoe/router.py#L67-L94) |
| 3.3.1 基于分布判别的阶段判别器 (DSD) | [router.py:97-133](models/vmoe/router.py#L97-L133) |
| 3.3.2 原型近邻插值特征增广 | [augment.py:5-45](models/vmoe/augment.py#L5-L45) |

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

* **CIFAR-100**：基类 60 / 增量类 5-way × 8 阶段
* **MiniImageNet**：基类 60 / 增量类 5-way × 8 阶段
* **CUB-200**：基类 100 / 增量类 10-way × 10 阶段

所有图像统一缩放至 $224\times224$，增广策略为随机裁剪 + 随机水平翻转 + 随机擦除。

### 3.3 训练超参（基类 / 新类共享骨架）

| 参数 | MiniImageNet | CIFAR-100 | CUB-200 |
| --- | --- | --- | --- |
| 优化器 | SGD (momentum 0.9, weight decay $5\!\times\!10^{-4}$) | 同左 | 同左 |
| 初始学习率 | 0.1 | 0.1 | 0.01 |
| 学习率衰减节点 | [40, 80] | [40, 80] | [30, 60, 90] |
| 基类训练轮数 | 100 | 100 | 120 |
| 新类训练轮数 | 20 | 20 | 20 |
| 批大小（基类） | 128 | 128 | 128 |
| 批大小（新类） | 全量 | 全量 | 全量 |
| 余弦温度 $\tau$ | 16 | 16 | 16 |

### 3.4 路由与增广超参

| 参数 | MSD | DSD |
| --- | --- | --- |
| 路由特征提取 | `frozen`（COCO 预训练 ResNet-18）或 `warp` | 同左 |
| 优化器（路由） | SGD | Adam(lr=$3\!\times\!10^{-3}$, wd=$1\!\times\!10^{-4}$) |
| 训练轮数（路由） | 5 | 500，Milestone @ [10, 20, 30] ×0.1 |
| 损失函数 | CE | MSE |
| 特征维度 | 512 | 512 |
| 瓶颈维度 | — | 256 |
| $K$（近邻数） | 5 | 10 |
| $N$（采样次数） | 5 | 5 |
| $\lambda$ 范围 | $[0.45, 0.75]$ | $[0.45, 0.75]$ |

正交损失系数：基类 $\lambda_\text{ortho}^\text{base}=0.01$，新类 $\lambda_\text{ortho}^\text{new}=0.05$，跨阶段 $\lambda_\text{cross}=0.8$。

### 3.5 启动方式

```bash
# 三个数据集的官方脚本入口
bash script/cifar100.sh
bash script/mini_imagenet.sh
bash script/cub200.sh
```

也可手动指定：

```bash
python train.py \
    -project vmoe \
    -dataset {cifar100|mini_imagenet|cub200} \
    -backbone_type clip_vit_b16 \
    -router_disc_type {msd|dsd} \
    -router_feat_mode {frozen|warp} \
    -lora_rank 8 -lora_alpha 16 \
    -lambda_cross 0.8 \
    -lambda_ortho_base 0.01 -lambda_ortho_new 0.05
```

### 3.6 推理流程

1. 阶段路由：图像经冻结 `ResNet-18`（或 `WaRP ResNet-18`）提取 512 维视觉特征，由 `MemoryBankSessionDiscriminator` 或 `DistributionSessionDiscriminator` 推断阶段 $t^*$。
2. 专家激活：在 CLIP ViT-B/16 的所有 FFN 中，将 $t^*$ 对应的 `SessionLoRAExpert` 切换为激活态，其余保持冻结。
3. 类别判别：768 维 `[CLS]` token 经 `SessionDecoupledClassifier` 选中第 $t^*$ 个子头，与历史所有阶段的权重拼接后计算余弦相似度，取 $\arg\max$。

---

## 4. 关键超参说明

| CLI 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-lora_rank` | 8 | 低秩分解的秩 $r$ |
| `-lora_alpha` | 16 | 专家输出缩放系数 $\alpha$（实际缩放为 $\alpha/r$） |
| `-lambda_cross` | 0.8 | 跨阶段正交系数 $\lambda_\text{cross}$（公式 3-6） |
| `-lambda_ortho_base` | 0.01 | 基类阶段正交损失权重 |
| `-lambda_ortho_new` | 0.05 | 增量阶段正交损失权重 |
| `-aug_lambda_min` | 0.45 | 插值强度下界 |
| `-aug_lambda_max` | 0.75 | 插值强度上界 |
| `-router_neighbor_k` | 5 (MSD) / 10 (DSD) | 原型近邻数 $K$ |
| `-router_sample_n` | 5 | 每个样本生成的偏移样本数 $N$ |
| `-router_loss_weight` | 1.0 | 路由损失在总损失中的权重 |
| `-router_bottleneck_dim` | 256 | DSD 中自动编码器的瓶颈维度 |
| `-temperature` | 16 | 分类余弦温度 $\tau$ |
| `-model_dir` | `None` | 预训练 checkpoint 路径；为空则使用 CLIP/ResNet-18 默认预训练权重 |

---

## 5. 复现检查清单

* [ ] 已安装 `transformers`、`torchvision`，并能加载 `openai/clip-vit-base-patch16`。
* [ ] 数据集已按 [data/](data/) 目录约定放置。
* [ ] GPU 显存 ≥ 12 GB（4 阶段批 128 训练时实测）。
* [ ] `script/*.sh` 顶部 `model_dir`、`router_disc_type`、`router_feat_mode` 已按实验矩阵设置。
* [ ] 检查点保存路径：`/data/xuzg/FSCIL/MoE-FSCIL/checkpoint/<dataset>/vmoe/<run_name>/`。

---

## 6. 引用方式

本方法包含三个相互独立的可消融模块，建议在论文中按以下方式报告消融结果：

| 缩写 | 含义 | 启用方式 |
| --- | --- | --- |
| LoRA-EM | 低秩专家模块 | 设置 `-lora_rank > 0` |
| SRM (MSD) | 基于记忆存储器的路由 | `-router_disc_type msd` |
| SRM (DSD) | 基于分布判别的路由 | `-router_disc_type dsd` |
| SDC | 阶段解耦分类 | 始终启用，可通过 `-lambda_ortho_*=0` 退化为单一分类头 |
| PNN-FA | 原型近邻插值 | `-router_neighbor_k > 0` 且 `-router_sample_n > 0` |

---

## 7. 许可

本仓库仅供学术研究使用，CLIP 与 ResNet-18 的预训练权重遵循其各自原始许可。
