# Precompute

`precompute/` contains every offline stage that must be finished before training or cache-based evaluation.

These stages are intentionally kept outside the main training loop for efficiency. The expensive feature extraction work is done once, saved to disk, and then reused by PAVAS.

## What PAVAS Expects

Before you launch `train.py` or `evaluate.py`, prepare all three of the following:

1. VGGSound feature caches
2. CLIP patch features
3. PPE outputs

## Recommended Order

Run the precompute stages in this order:

1. [mmaudio_cache/README.md](mmaudio_cache/README.md)
2. [clip_patches/README.md](clip_patches/README.md)
3. [ppe/README.md](ppe/README.md)

That order matters:

- the VGGSound feature caches are the backbone-side offline inputs
- the CLIP patch features are the object-branch visual inputs
- the PPE stage adds mass, segmentation, moving-object, and velocity signals

## Precompute Categories

### 1. VGGSound Feature Caches

This stage follows the upstream MMAudio cache extraction pipeline and produces the memmaps and TSV files consumed by PAVAS training and cache-based evaluation.

Guide:

- [mmaudio_cache/README.md](mmaudio_cache/README.md)

### 2. CLIP Patch Features

This stage extracts patch-level visual features for the PAVAS object-conditioning branch.

Guide:

- [clip_patches/README.md](clip_patches/README.md)

### 3. PPE Outputs

This stage produces:

- moving-object discovery outputs
- mass estimates
- segmentation masks and prompts
- velocity estimates

Use `precompute/ppe/scripts/run_ppe_pipeline.py` to execute these stages
for each dataset split rather than looping over videos and prompts manually.

Guide:

- [ppe/README.md](ppe/README.md)

## Minimal Checklist

If you are setting up PAVAS for the first time, the practical order is:

1. activate `.venv`
2. prepare VGGSound locally
3. prepare the MMAudio-side external weights
4. generate VGGSound feature caches
5. extract CLIP patch features
6. run the PPE stages
7. update config paths
8. launch training or evaluation
