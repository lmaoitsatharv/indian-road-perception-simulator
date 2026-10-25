# Road model architecture and training

The real-road segmenter is a U-Net-style decoder with a MobileNetV2 ImageNet encoder. This document describes its data contract, training stages and locked evaluation. The separate detector and synthetic-road model are summarized in [results](docs/RESULTS.md).

## Current model and evaluation

The default for real photographs and videos is the morning-best trial-13 checkpoint. Its validation road IoU is 92.1258%. It was selected using development data, then its checkpoint, preprocessing and threshold were locked before test-image evaluation. On all 689 real test images it scored **91.6549% road IoU and 95.6457% Dice**.

The original and boundary checkpoints remain validation references; they were not separately evaluated on that test set. See [model metadata](models/README.md), `models/retained_models.json`, `models/segmentation_final_lock.json` and [portable segmentation results](docs/results/segmentation.json). Checkpoint binaries and full experiment directories are supplied separately from source.

## Input contract

`idd_data.py` loads the retained 5,507 training / 687 validation / 689 test pairs. Images are float32 RGB at 512x512 and normalized exactly once to [-1,1]. Targets are float32 `[512,512,1]` with labels exactly 0/1. The network performs no second normalization or resizing. See [preprocessing](PREPROCESSING.md).

Masks must not be divided by 255: their positive label is already 1. Display-only 0/255 previews are separate artifacts. Training can flip image and mask together; validation and test receive no augmentation.

## Encoder

The pretrained MobileNetV2 classification head is removed with `include_top=False`. The convolutional filters supply features at five scales:

| Encoder output | Spatial size | Channels |
| --- | --- | --- |
| `block_1_expand_relu` | 256x256 | 96 |
| `block_3_expand_relu` | 128x128 | 144 |
| `block_6_expand_relu` | 64x64 | 192 |
| `block_13_expand_relu` | 32x32 | 576 |
| `block_16_project` | 16x16 | 320 |

These are intermediate feature maps, not five road predictions. The original weight archive's 224 naming does not require resizing project inputs to 224.

## Decoder and output

Four decoder blocks use 256, 128, 64 and 32 filters. Each upsamples, concatenates the matching encoder feature and refines it with convolutions. These skip connections preserve spatial detail without exposing target labels to the network. A final upsample and 16-filter refinement restore 512x512 resolution. A 1x1 sigmoid head produces one road probability per pixel.

The initial network had 6,202,337 parameters; 4,357,473 were trainable while the encoder was frozen. Encoder BatchNorm remains in inference mode. During later fine-tuning, selected convolutional layers can be trainable while their BatchNorm layers remain fixed.

## Objective and metrics

The road objective is equal-weight per-image binary cross-entropy plus one-minus-soft-Dice:

```text
loss = mean per-image BCE + mean per-image (1 - soft Dice)
soft Dice = (2*sum(truth*probability) + 1e-6)
            / (sum(truth) + sum(probability) + 1e-6)
```

Soft probabilities remain differentiable; predictions are not thresholded before loss calculation. Adam calculates updates from backpropagated gradients. At evaluation threshold 0.5, metrics accumulate road TP/FP/FN over all pixels and batches:

```text
road IoU  = TP / (TP + FP + FN)
road Dice = 2*TP / (2*TP + FP + FN)
```

This is global class-1 overlap, not per-image IoU averaging or a road/background mean. Global empty/empty overlap is defined as 1. Boundary metrics use separately declared pixel bands. Pixel accuracy alone can hide poor road overlap when background is common.

## Recorded training stages

| Stage | Configuration | Outcome |
| --- | --- | --- |
| Initial frozen encoder | 512x512, batch 4, seed 42, Adam 0.001, clipnorm 1, maximum 20 epochs | 18 completed epochs; original best epoch 14, validation road IoU 92.0345% |
| Selected trial 13 | Original trained checkpoint, shadow augmentation, encoder blocks 13-16 except BatchNorm, fixed decoder BatchNorm, batch 16, LR 0.000003 | Five completed epochs; selected epoch 1, validation road IoU 92.1258% |
| Boundary experiment | Separate training-only boundary emphasis and declared edge metrics | Retained as a validation reference; not the default real model |

The first phase used early stopping on validation road IoU with patience 5 and minimum change 0.001. Validation-loss stagnation halved learning rate after two epochs, down to 1e-6. A selected later epoch number is relative to that experiment, not evidence that the full model learned from scratch in one epoch.

The first run took about 88.1 minutes in the reference RTX 4060 Laptop / WSL2 environment, using TensorFlow 2.20.0 and Keras 3.15.1. Timings describe that workload and machine. Initial verification established finite learning signals, changed decoder weights and unchanged frozen encoder weights; its discarded probe was not training initialization.

## Starting a separate experiment

Prepare the runtime, data and trusted encoder/checkpoint artifacts as described in [getting started](docs/GETTING_STARTED.md) and [WSL setup](WSL_SETUP.md). The explicit entry point is:

```bash
python train_idd.py --run-dir runs/frozen_NEW --epochs 20 --batch-size 4
```

Omitting `--initial-model` creates a fresh decoder with the cached ImageNet encoder. Fine-tuning requires an explicit trusted checkpoint and phase. Use fresh output names: the trainer refuses to overwrite an existing run. It does not silently fall back to CPU unless CPU execution is explicitly selected.

A run saves best/last checkpoints, per-epoch CSV/JSON history, actual configuration and completion/failure state. The retained split manifest is hashed before and after training. Historical full evidence remains under `runs/frozen_01/`, `runs/overnight_20261003/` and `runs/boundary_20261003/`; these local artifacts are outside the source repository.

## Limits and model retention

The source copy's exact original class/ignore mapping is not fully established. Its 110 all-zero-mask pairs were excluded from prepared copies with originals and removal records preserved. That does not prove every excluded mask was semantically erroneous. Original drive identities are absent, so the split cannot establish sequence-independent generalization despite exact-image checks.

The completed 689-image real test is now inspected and must not be reused to select a new model while claiming an untouched score. Further real-road claims require independent evaluation data. Repeated reviewed-video examples are development guards.

Synthetic adaptation and the newer simulation-road model have explicit separate scopes. Earlier adaptation candidates failed strict real road/boundary/video retention checks. Synthetic gains do not replace the locked real model; see [real-model retention](docs/REAL_MODEL_RETENTION.md).

Core implementation: `idd_model.py`, `idd_data.py`, `train_idd.py`, `idd_experiments.py` and `idd_boundary.py`. The [code map](docs/CODE_MAP.md) identifies supporting verification/reporting utilities.
