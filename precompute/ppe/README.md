# PPE

`precompute/ppe/` contains the Physics Parameter Estimator stage used by PAVAS.

PPE produces the structured object-level signals consumed by PAVAS:

- moving-object discovery outputs
- mass estimates
- segmentation masks and prompts
- per-object velocity estimates

The PPE implementation is vendored in this repository, so the default path does not require a separate checkout of Qwen2.5-VL, Grounded-SAM-2, or CUT3R.

## Run A Dataset Split

The recommended entrypoint runs all PPE stages in order, including 8 fps
frame extraction and CUT3R for every segmentation prompt:

```bash
source .venv/bin/activate
python precompute/ppe/scripts/download_ppe_assets.py
python precompute/ppe/scripts/run_ppe_pipeline.py \
  --tsv sets/vgg-train.tsv \
  --video-root /path/to/vggsound/video \
  --output-root /path/to/ppe
```

Run the same command with `sets/vgg-val.tsv` and `sets/vgg-test.tsv` to
prepare those splits. Each TSV needs an `id` column, and each video must be
available as `--video-root/<id>.mp4`. Run from the repository root.

The standalone PPE runner defaults to eight seconds for its frame-based
segmentation and velocity stages, matching the VGGSound training setup. Qwen
discovery and mass estimation still inspect the supplied video. For a different
frame-stage duration, specify both the duration and the expected 8 fps frame
count (for example,
`--duration 12 --frame-count 96`). For long videos,
`--discovery-timeout` can increase the Qwen per-video inference limit from its
12-second default.

For a one-video trial, append `--limit 1`. To preview which videos a shard
would process without loading any models, append `--dry-run`. For parallel
execution, launch one process per GPU with the same `--num-shards N` and a
distinct `--shard-index` from `0` to `N-1`:

```bash
python precompute/ppe/scripts/run_ppe_pipeline.py \
  --tsv sets/vgg-train.tsv \
  --video-root /path/to/vggsound/video \
  --output-root /path/to/ppe \
  --num-shards 4 --shard-index 0
```

The script reuses valid existing outputs on rerun. It fingerprints each source
video and records the frame-stage duration and frame count. Changing the video
or frame settings invalidates the affected downstream outputs; stale
segmentation and velocity directories are cleared before recomputation. It
reports videos with no
discovered moving object as `no_moving_objects`, and records missing or
invalid outputs as `failed` in
`/path/to/ppe/work/<tsv-name>/shard_0000/summary.json`. A shard returns a
nonzero exit code if any video failed; rerun the same command to retry those
outputs. `--force` clears and recomputes all PPE outputs for the selected
videos. The
intermediate moving outputs and extracted frames stay under `work/`.
For a valid no-moving-object result, the runner writes
`velocity/<video_id>/no_moving_objects.json`. This records an intentional empty
result separately from missing or failed PPE outputs.

Set the corresponding roots in `configs/data/base.yaml` to
`/path/to/ppe/mass`, `/path/to/ppe/segmentation`, and
`/path/to/ppe/velocity`. The script loads Qwen, Florence2/SAM2, and CUT3R
on a GPU. One process per GPU is recommended.

The commands below describe the individual wrapper stages. You do not need
to run them separately after a successful dataset-split run.

## Before You Run PPE

Prepare the following first for either the dataset-split runner or individual wrappers:

1. activate `.venv`
2. prepare the VGGSound videos locally
3. download the public PPE assets and checkpoints

### 1. Activate The Environment

```bash
source .venv/bin/activate
```

### 2. Point To Your VGGSound Videos

```bash
export PAVAS_VGGSOUND_VIDEO_DIR=/path/to/vggsound/video
```

### 3. Download The Public PPE Assets

```bash
python precompute/ppe/scripts/download_ppe_assets.py
```

This script downloads or prefetches:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `microsoft/Florence-2-large`
- `sam2.1_hiera_large.pt`
- `cut3r_512_dpt_4_64.pth`

Qwen and Florence are stored in the default Hugging Face cache used by their
`from_pretrained` calls. To use a custom cache, set `HF_HOME` before both
downloading and running PPE.

Default checkpoint locations:

- `precompute/ppe/grounded_sam2/checkpoints/sam2.1_hiera_large.pt`
- `precompute/ppe/cut3r/src/cut3r_512_dpt_4_64.pth`

Optional overrides:

```bash
export PAVAS_GSAM2_CKPT=/path/to/sam2.1_hiera_large.pt
export PAVAS_CUT3R_MODEL_PATH=/path/to/cut3r_512_dpt_4_64.pth
```

## PPE Stage Order

Run PPE in the following order:

1. moving-object discovery
2. mass estimation
3. segmentation
4. velocity estimation
5. output validation

The wrapper entrypoints are:

- `precompute/ppe/scripts/run_moving_object_discovery.py`
- `precompute/ppe/scripts/run_mass_estimation.py`
- `precompute/ppe/scripts/run_segmentation.py`
- `precompute/ppe/scripts/run_velocity_estimation.py`
- `precompute/ppe/scripts/verify_ppe_outputs.py`

## Step 1. Moving-Object Discovery

This stage reads videos and produces the moving-object JSON files used by later PPE stages.
Create a UTF-8 list with one video filename per line, relative to
`PAVAS_VGGSOUND_VIDEO_DIR` (or use absolute video paths). For example:

```text
VIDEO_ID.mp4
```

Example:

```bash
python precompute/ppe/scripts/run_moving_object_discovery.py \
  --list-file /path/to/video_list.txt \
  --out-root-base /path/to/ppe/moving
```

Default output: `/path/to/ppe/moving/<video_id>/physics_moving.json`.
Pass `--out-subdir split_00_json` when processing a shard separately.

## Step 2. Mass Estimation

This stage reads the moving-object outputs and produces `mass_estimation.json`.

Example:

```bash
python precompute/ppe/scripts/run_mass_estimation.py \
  --in-root-base /path/to/ppe/moving \
  --out-root-base /path/to/ppe/mass
```

Default output: `/path/to/ppe/mass/<video_id>/mass_estimation.json`.
For sharded moving outputs, mass estimation keeps the input shard name by default.

If `--in-root-base` contains a single moving-output subdirectory, `run_mass_estimation.py` can infer it automatically. If multiple subdirectories exist, pass `--in-subdir` explicitly.

## Step 3. Segmentation

This stage reads the moving-object outputs together with extracted frames and produces segmentation masks plus prompt files.
Extract frames at 8 fps into one directory per video before running it:

```bash
mkdir -p /path/to/video_frames/VIDEO_ID
ffmpeg -i /path/to/vggsound/video/VIDEO_ID.mp4 -t 8 -vf fps=8 \
  /path/to/video_frames/VIDEO_ID/%06d.jpg
```

Example:

```bash
python precompute/ppe/scripts/run_segmentation.py \
  --moving-root /path/to/ppe/moving \
  --manifest-out /path/to/ppe/manifests \
  --frames-root /path/to/video_frames \
  --output-root /path/to/ppe/segmentation
```

Typical output:

- `/path/to/ppe/segmentation/<video_id>/<prompt_id>/FL2_mask_data/mask_*.npz`
- `/path/to/ppe/segmentation/<video_id>/<prompt_id>/FL2_text_prompt/prompt.json`

If you pass `--split-index 0`, the outputs are placed under
`/path/to/ppe/segmentation/split_00/`. Use that directory in the velocity
command, or use the parent root with the output validator.

## Step 4. Velocity Estimation

This stage reads one frame sequence, one segmentation directory, and one prompt directory, then produces `instant_speeds.csv`.

Example:

```bash
python precompute/ppe/scripts/run_velocity_estimation.py \
  --seq-path /path/to/video_frames/VIDEO_ID \
  --mask-path /path/to/ppe/segmentation/VIDEO_ID/PROMPT_ID/FL2_mask_data \
  --prompt-path /path/to/ppe/segmentation/VIDEO_ID/PROMPT_ID/FL2_text_prompt \
  --output-dir /path/to/ppe/velocity/VIDEO_ID \
  --prompt-id PROMPT_ID
```

Typical output:

- `/path/to/ppe/velocity/VIDEO_ID/PROMPT_ID/speeds/instant_speeds.csv`

Run this command for each prompt directory produced by segmentation. The
`--fps` value must match the frame extraction and segmentation FPS.

## Step 5. Validate The Outputs

After all four PPE stages finish, run the validator on one sample first:

```bash
python precompute/ppe/scripts/verify_ppe_outputs.py \
  --video-id VIDEO_ID \
  --prompt-id PROMPT_ID \
  --mass-root /path/to/ppe/mass \
  --velocity-root /path/to/ppe/velocity \
  --seg-root /path/to/ppe/segmentation
```

## Expected Final Layout

PAVAS expects a directory layout that contains:

- `mass_root/<video_id>/mass_estimation.json` as the preferred public layout
- `mass_root/<subdir>/<video_id>/mass_estimation.json` is also supported for legacy split-based runs
- `velocity_root/<video_id>/<prompt_id>/speeds/instant_speeds.csv`
- `velocity_root/<video_id>/no_moving_objects.json` for videos with no discovered moving object
- `seg_root/<video_id>/<prompt_id>/FL2_text_prompt/prompt.json`
- `seg_root/<video_id>/<prompt_id>/FL2_mask_data/mask_*.npz`

## Vendored Components

The PPE implementation included here is organized as:

- `qwen2_5_vl/` for moving-object discovery and mass estimation
- `grounded_sam2/` for segmentation
- `cut3r/` for velocity estimation

The wrappers in `precompute/ppe/scripts/` standardize the inputs and outputs expected by PAVAS.
