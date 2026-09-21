# CLIP Patch Features

This stage extracts the patch-level visual features used by the PAVAS object-conditioning branch.

These patch features are saved offline and reused during training and cache-based evaluation.

## What This Stage Produces

The output of this stage is a saved patch grid that PAVAS combines with:

- segmentation masks
- mass estimates
- velocity estimates

to build object-wise conditioning features before they are injected into the generation backbone.

## Why This Stage Is Offline

Patch extraction across the full dataset is expensive. Running it once offline is much faster and more reproducible than recomputing it inside every training run.

## Entry Points

The repository provides:

- [extract_clip_patches.py](extract_clip_patches.py)
- [extract_clip_patches_mp.py](extract_clip_patches_mp.py)

The multi-process variant is typically the more practical choice for large-scale extraction.
It also works directly on one GPU without a distributed launcher.

## Typical Workflow

1. activate `.venv`
2. make sure VGGSound videos are available locally
3. use the provided `sets/vgg-{train,val,test}.tsv` lists
4. extract each split into a different output directory
5. point `configs/data/base.yaml` to the generated TSV and NPY for each split

## Example

After activating `.venv`, extract each split. `--video_root` must contain
`{id}.mp4` files for the IDs in the TSV:

```bash
for split in train val test; do
  python precompute/clip_patches/extract_clip_patches_mp.py \
    --video_root /path/to/vggsound/videos \
    --tsv "sets/vgg-${split}.tsv" \
    --output_dir "/path/to/clip_patches/${split}" \
    --clip_model vit_b_16 \
    --clip_size 224 \
    --batch_size 8 \
    --num_workers 8
done
```

This uses [LAION CLIP ViT-B/16](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K).
Its pre-projection patch tokens have shape `(T, 14, 14, 768)` per video;
PAVAS expects the final dimension to be 768. Do not use ViT-H/384 patch
caches with this configuration.

Each run produces `vgg-patches.tsv` and
`vgg-patches/clip_patches.npy` inside its output directory. The generated
TSV records the row order of the NPY; **do not** point the loader to the
input `sets/vgg-*.tsv`. Configure, for example, the training split as:

```yaml
clip_patches:
  tsv: /path/to/clip_patches/train/vgg-patches.tsv
  npy: /path/to/clip_patches/train/vgg-patches/clip_patches.npy
```

Use the corresponding `val/` and `test/` paths for those splits. The loader
supports this NPY format directly; no flattening conversion is required.

If you already have patch shards and only want to merge them:

```bash
python precompute/clip_patches/extract_clip_patches_mp.py \
  --output_dir /path/to/clip_patches/train \
  --merge_only
```

## Before You Continue

Before moving on to PPE or training, confirm that:

- both output files exist for every required split
- the configured TSV and NPY for each split come from the same run
- the extraction settings match the feature format expected by your run
