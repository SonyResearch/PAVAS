# Training

This document assumes you are starting after the environment and offline precompute stages are already finished.

## Before You Launch Training

Confirm that all of the following are ready:

1. VGGSound is available locally
2. VGGSound feature caches are prepared
3. CLIP patch features are prepared
4. PPE outputs are prepared
5. external weights are available
6. an MMAudio backbone initialization checkpoint is available

For the VGGSound feature-cache stage, follow:

- https://github.com/hkchengrex/MMAudio/blob/main/docs/TRAINING.md

## Recommended Preparation Order

The practical order is:

1. prepare the dataset locally
2. prepare VGGSound feature caches
3. prepare CLIP patch features
4. prepare PPE outputs
5. verify config paths
6. launch training

## Main Config

Start from:

- `configs/train/finetune.yaml`

Before launching a real run, review:

- dataset and cache paths
- PPE roots
- patch feature roots
- checkpoint path
- output directory
- model size

The active data and checkpoint paths are in `configs/data/base.yaml` and
`configs/base_config.yaml`. `configs/paths.example.yaml` is not loaded.
The CLIP patch `tsv` must be the index written alongside the configured
`npy`, not the input `sets/vgg-*.tsv`.

`finetune.yaml` sets the global batch size to 512; `train.py` divides this
by the number of GPUs. Override `batch_size` if it exceeds GPU memory.

## Single-GPU Example

```bash
source .venv/bin/activate
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 \
  train.py exp_id=pavas_finetune model=small_16k batch_size=4 \
  weights=/path/to/mmaudio_small_16k.pth
```

## Multi-GPU Example

```bash
source .venv/bin/activate
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 \
  train.py exp_id=pavas_finetune model=small_16k batch_size=8 \
  weights=/path/to/mmaudio_small_16k.pth
```

`train.py` loads `configs/train/finetune.yaml` by default. Use `weights=` for
backbone initialization; `checkpoint=` is for resuming a PAVAS training run.

## Final Checklist

Before starting a long run, confirm:

1. the config paths point to your local dataset, caches, patch features, PPE outputs, and weights
2. the checkpoint path exists
3. the output directory is writable
4. the smoke tests passed earlier
