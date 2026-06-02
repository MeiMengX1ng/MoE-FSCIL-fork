# VMoE-FSCIL

PyTorch implementation of a few-shot class-incremental learning method based on:

`visual feature-based session routing + session-specific LoRA experts + session-decoupled classifiers + prototype-neighbor feature interpolation`

This repository is no longer documented as the original CEC project. The current README only describes the present `vmoe` algorithm implementation in this workspace.

## Overview

The current method targets FSCIL with the following design:

1. A frozen visual backbone is used as the main classifier feature extractor.
2. Session-specific LoRA experts are attached to the backbone so that each incremental stage learns its own parameter-efficient adaptation.
3. A session routing module predicts which stage an input sample belongs to.
4. A session-decoupled classifier maintains separate classifier parameters for different stages.
5. A prototype-nearest-neighbor interpolation module augments sparse incremental-stage features.

The code path for this implementation is centered on:

- [train.py](/d:/Project/MoE-FSCIL/train.py:1)
- [models/vmoe/Network.py](/d:/Project/MoE-FSCIL/models/vmoe/Network.py:1)
- [models/vmoe/fscil_trainer.py](/d:/Project/MoE-FSCIL/models/vmoe/fscil_trainer.py:1)

## Implemented Architecture

The current implementation consists of the following modules.

### 1. Backbone

Implemented in [models/vmoe/Network.py](/d:/Project/MoE-FSCIL/models/vmoe/Network.py:1).

Supported backbone types:

- `clip_vit_b16`
- `resnet18`

Current intended default is `clip_vit_b16`.

Backbone behavior:

- Input images are internally resized to `224 x 224`.
- Backbone parameters are frozen by default.
- If `-backbone_model_dir` is provided, backbone weights are loaded with `strict=False`.

For the `ViT` path:

- The model uses `torchvision.models.vit_b_16(weights=None)`.
- The final `class token` is used as the classification feature.
- LoRA is injected into `encoder.layers`, not into the `heads` section.

### 2. LoRA Experts

Implemented in [models/vmoe/lora.py](/d:/Project/MoE-FSCIL/models/vmoe/lora.py:1).

Core classes:

- `SessionLoRAExpert`
- `MultiSessionLoRA`
- `LoRALinear`

Design:

- Each session owns an independent low-rank expert.
- For a linear layer, the effective output is:

```text
y = W0 x + LoRA_session(x)
```

- The low-rank update is parameterized by session-specific `A` and `B`.
- The scaling factor is `alpha / rank`.

Current implementation detail:

- For `clip_vit_b16`, LoRA is injected into all `nn.Linear` layers under `backbone.encoder.layers`.
- `heads` are explicitly excluded by structure.
- For the fallback `resnet18` path, a feature-level external LoRA residual is still kept as a compatibility fallback.

### 3. Session Routing Module

Implemented in [models/vmoe/router.py](/d:/Project/MoE-FSCIL/models/vmoe/router.py:1).

The routing module contains:

- a routing feature extractor
- a session discriminator

Two routing feature extractor modes are supported:

- `frozen`
- `warp`

#### Frozen Mode

`FrozenResNet18Extractor`

- Uses `resnet18`
- Extracts route features from a convolutional feature map
- Applies adaptive average pooling to obtain a `512`-dimensional route feature
- Keeps extractor parameters frozen

#### WaRP Mode

`WaRPResNet18Extractor`

- Uses `resnet18`
- Replaces convolution layers with WaRP-enabled rotated modules
- Supports orthonormal basis computation and weight restoration

WaRP-related code is implemented in:

- [models/vmoe/warp.py](/d:/Project/MoE-FSCIL/models/vmoe/warp.py:1)

### 4. Session Discriminators

Two routing discriminators are implemented.

#### MSD: Memory Bank-based Session Discriminator

Implemented in [models/vmoe/router.py](/d:/Project/MoE-FSCIL/models/vmoe/router.py:43).

Behavior:

- Stores class prototypes from previously seen sessions
- Fuses current route feature with pooled historical memory
- Predicts session logits through an MLP projector

Important implementation note:

- `memory_bank` is a Python dictionary
- It is not part of the standard `state_dict`
- It is saved separately through `extra_state` in checkpoints

#### DSD: Distribution-based Session Discriminator

Implemented in [models/vmoe/router.py](/d:/Project/MoE-FSCIL/models/vmoe/router.py:76).

Behavior:

- Maintains one autoencoder per session
- Uses reconstruction error as routing score
- Predicts the session with minimum reconstruction error

### 5. Session-Decoupled Classifier

Implemented in [models/vmoe/classifier.py](/d:/Project/MoE-FSCIL/models/vmoe/classifier.py:1).

Behavior:

- One classifier head per session
- Base session uses the first `base_class` rows
- Incremental sessions use disjoint class ranges of length `way`

At inference:

- The effective classifier weight matrix is assembled from:
  - base session weights
  - current incremental session ranges

The classifier also implements feature-space orthogonality regularization:

- intra-session orthogonality
- cross-session orthogonality

### 6. Prototype-Neighbor Feature Augmentation

Implemented in [models/vmoe/augment.py](/d:/Project/MoE-FSCIL/models/vmoe/augment.py:1).

Behavior:

- Builds per-class prototypes from current batch/session features
- Finds nearest neighbor prototypes by cosine similarity
- Creates shifted features using:

```text
f_tilde = f + lambda * delta
delta = 0.5 * (center - neighbor)
```

Current implementation notes:

- `lambda` is sampled from `[aug_lambda_min, aug_lambda_max]`
- `k` nearest neighbors and `sample_n` sampling rounds are configurable
- The sign of `delta` currently follows the written formula in the algorithm text

## Training Flow

Training logic is implemented in:

- [models/vmoe/fscil_trainer.py](/d:/Project/MoE-FSCIL/models/vmoe/fscil_trainer.py:1)
- [models/vmoe/helper.py](/d:/Project/MoE-FSCIL/models/vmoe/helper.py:1)

For each session:

1. Set `active_session`
2. Build session dataloaders
3. If `router_feat_mode=warp`, compute the WaRP orthonormal basis
4. Optimize:
   - current session LoRA parameters
   - current session classifier head
   - routing discriminator parameters
   - routing feature extractor parameters if `warp` is enabled
5. Evaluate on all seen classes using routing-based inference
6. Update memory bank if `MSD` is used
7. Save checkpoint

The loss currently combines:

- classification cross-entropy
- orthogonality loss
- routing loss

Routing loss:

- `MSD`: cross-entropy over session labels
- `DSD`: reconstruction MSE for the current session autoencoder

## Inference Flow

At inference time the model follows this order:

1. Extract routing features
2. Predict session id with `SRM`
3. Activate the corresponding session-specific LoRA path
4. Apply the corresponding effective classifier
5. Output logits over seen classes

The forward path is implemented in [models/vmoe/Network.py](/d:/Project/MoE-FSCIL/models/vmoe/Network.py:1).

Important detail:

- Inference is routed by default
- The model supports mixed-session batches
- Internally, per-sample session predictions are handled and then scattered into a global logit tensor

## Code Structure

Current `vmoe` files:

- [models/vmoe/Network.py](/d:/Project/MoE-FSCIL/models/vmoe/Network.py:1): top-level model assembly
- [models/vmoe/fscil_trainer.py](/d:/Project/MoE-FSCIL/models/vmoe/fscil_trainer.py:1): training loop, checkpointing, session lifecycle
- [models/vmoe/helper.py](/d:/Project/MoE-FSCIL/models/vmoe/helper.py:1): per-epoch train/eval utilities
- [models/vmoe/lora.py](/d:/Project/MoE-FSCIL/models/vmoe/lora.py:1): LoRA experts and linear-layer injection
- [models/vmoe/router.py](/d:/Project/MoE-FSCIL/models/vmoe/router.py:1): routing feature extractor and discriminators
- [models/vmoe/classifier.py](/d:/Project/MoE-FSCIL/models/vmoe/classifier.py:1): session-decoupled classifier and orthogonality loss
- [models/vmoe/augment.py](/d:/Project/MoE-FSCIL/models/vmoe/augment.py:1): prototype-based feature interpolation
- [models/vmoe/warp.py](/d:/Project/MoE-FSCIL/models/vmoe/warp.py:1): WaRP-like rotated convolution support

Training scripts:

- [script/cifar100.sh](/d:/Project/MoE-FSCIL/script/cifar100.sh:1)
- [script/mini_imagenet.sh](/d:/Project/MoE-FSCIL/script/mini_imagenet.sh:1)
- [script/cub200.sh](/d:/Project/MoE-FSCIL/script/cub200.sh:1)

## Command-Line Arguments

The main CLI is defined in [train.py](/d:/Project/MoE-FSCIL/train.py:1).

Important arguments for the current algorithm:

### General

- `-project vmoe`
- `-dataset {cifar100, mini_imagenet, cub200}`
- `-gpu`
- `-seed`

### Backbone

- `-backbone_type {clip_vit_b16, resnet18}`
- `-backbone_feat_dim`
- `-model_image_size`
- `-backbone_model_dir`

### Routing

- `-router_disc_type {msd, dsd}`
- `-router_feat_mode {frozen, warp}`
- `-router_bottleneck_dim`
- `-router_loss_weight`
- `-router_model_dir`

### LoRA and classifier

- `-lora_rank`
- `-lora_alpha`
- `-lambda_cross`
- `-lambda_ortho`
- `-temperature`

### Feature augmentation

- `-aug_lambda_min`
- `-aug_lambda_max`
- `-router_neighbor_k`
- `-router_sample_n`

### Optimization

- `-epochs_base`
- `-epochs_new`
- `-lr_base`
- `-lr_new`
- `-milestones`
- `-milestones_new`
- `-gamma`
- `-decay`
- `-momentum`

## Training Scripts

The repository includes shell scripts modeled after the reference script style:

- [script/cifar100.sh](/d:/Project/MoE-FSCIL/script/cifar100.sh:1)
- [script/mini_imagenet.sh](/d:/Project/MoE-FSCIL/script/mini_imagenet.sh:1)
- [script/cub200.sh](/d:/Project/MoE-FSCIL/script/cub200.sh:1)

Each script exposes the following top-level knobs:

- `router_disc_type`
- `router_feat_mode`
- `router_neighbor_k`
- `router_sample_n`
- `backbone_model_dir`
- `router_model_dir`

Recommended defaults currently encoded in the scripts:

- `MSD`: `router_disc_type=msd`, `k=5`
- `DSD`: manually switch `router_disc_type=dsd`, `k=10`
- `frozen`: use when reproducing the frozen routing extractor variant
- `warp`: use when testing the WaRP routing extractor variant

## Checkpoints

Checkpoints are saved under:

```text
checkpoint/<dataset>/vmoe/<mode>/<schedule>/
```

Checkpoint contents:

- `params`: model parameters
- `extra_state.active_session`
- `extra_state.memory_bank` for `MSD`

This is necessary because `memory_bank` is not a registered parameter or buffer.

## Pretrained Weights

Current code supports external weight injection via:

- `-backbone_model_dir`
- `-router_model_dir`
- `-model_dir` for resumed full checkpoints

These paths are expected to be prepared on the server side.

## Device Support

Current implementation is designed primarily for GPU training.

Important note:

- The code still contains multiple CUDA-specific paths
- The implementation is not yet fully device-agnostic
- Full local CPU training is not the intended mode at this stage

It can be made CPU-compatible later, but that is not the current default behavior.

## Current Implementation Boundaries

The current code is a working implementation scaffold of the target algorithm, but some points should be treated as explicit boundaries:

1. `ViT` LoRA injection is currently implemented over all linear layers inside `encoder.layers`.
   It is not yet narrowed further to only attention/MLP sublayers.
2. The code relies on `torchvision`'s `vit_b_16` internal structure.
   If the server environment uses a different version, minor structural adaptation may be needed.
3. `MSD` and `DSD` are implemented and trainable, but they are still engineering realizations of the algorithm description rather than a fully benchmark-validated reproduction.
4. The feature interpolation direction currently follows the written formula in the provided algorithm text.
   If the final intended direction is "toward the neighbor prototype" rather than the current sign convention, this part should be revised explicitly.
5. Root-level legacy modules such as `models/base` and `models/cec` still physically exist in the repository for code dependency reasons, but they are not the documented target method anymore.

## Recommended Usage

For the current repository state, the recommended way to use the algorithm is:

1. Prepare datasets in the existing FSCIL folder layout.
2. Prepare server-side pretrained weights for the main backbone and, if needed, the routing extractor.
3. Start from one of:
   - [script/cifar100.sh](/d:/Project/MoE-FSCIL/script/cifar100.sh:1)
   - [script/mini_imagenet.sh](/d:/Project/MoE-FSCIL/script/mini_imagenet.sh:1)
   - [script/cub200.sh](/d:/Project/MoE-FSCIL/script/cub200.sh:1)
4. Choose one routing configuration:
   - `MSD + frozen`
   - `MSD + warp`
   - `DSD + frozen`
   - `DSD + warp`
5. Launch training on a GPU server.

## Repository Status

This repository is currently documented as an implementation workspace for the present `VMoE-FSCIL` algorithm only.

The README intentionally does not preserve the original CEC project description.
