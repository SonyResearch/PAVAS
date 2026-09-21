# Installation

This document is written for a first-time user starting from an empty environment.
Python 3.10+ and an NVIDIA GPU with a compatible CUDA runtime are required for
the training and PPE model stages.

## Step 1. Create The Python Environment

The easiest path is:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

If you want to install the environment manually, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install nvidia-cublas-cu12==12.4.5.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec==0.1.1
pip install open_clip_torch==2.29.0 "numpy<2.1" "huggingface-hub<1.0" "transformers==4.51.3" "tokenizers==0.21.1"
pip install accelerate "qwen-vl-utils[decord]" "huggingface-hub<1.0" opencv-python-headless supervision imageio imageio-ffmpeg viser roma matplotlib tqdm scipy einops trimesh tensorboard "pyglet<2" lpips h5py scikit-learn gdown
pip install -e .
pip install 'git+https://github.com/hkchengrex/av-benchmark.git'
```

Validated versions:

- `torch==2.5.1`
- `torchcodec==0.1.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `open_clip_torch==2.29.0`
- `transformers==4.51.3`
- `numpy<2.1`
- `huggingface-hub<1.0`

The CUDA build downloads several large wheels. If the default temporary
filesystem is small, set `TMPDIR` to a location with enough free space before
running the setup script, for example `TMPDIR=/path/to/large/tmp bash scripts/setup_env.sh`.

## Step 2. Make FFmpeg Available

PAVAS uses `torio` for video I/O. If `torio` cannot find a compatible FFmpeg runtime, install:

```bash
conda install -c conda-forge 'ffmpeg<7'
```

The PPE batch runner uses the `imageio-ffmpeg` executable installed by
`scripts/setup_env.sh` to extract frames. This bundled executable does not
replace the compatible FFmpeg runtime required by `torio`.

## Step 3. Run The Smoke Tests

After the environment is created, confirm that the repository imports correctly:

```bash
source .venv/bin/activate
bash scripts/run_smoke_tests.sh
```

This checks:

- Python imports
- config loading
- PPE input validation
- model sequence configuration

The smoke tests check imports and input contracts; they do not run the large
PPE models or generate audio.

## Step 4. Download VGGSound

Download and prepare VGGSound locally.

You will need:

- the original videos or an equivalent local copy
- the ability to generate MMAudio-style VGGSound feature caches
- the ability to extract frame sequences for PPE and CLIP patch extraction

This repository assumes VGGSound already exists on disk before you continue.

## Step 5. Prepare External Weights

Prepare the following before training or evaluation:

- VAE checkpoints
- BigVGAN vocoder checkpoint
- Synchformer checkpoint
- a backbone initialization checkpoint, for example an MMAudio checkpoint from the official MMAudio model releases

Create `ext_weights/` at the repository root and put these model assets there
under the exact filenames shown:

| File | Purpose |
| --- | --- |
| `empty_string.pth` | unconditional text embedding; required for both training and evaluation |
| `v1-16.pth` | 16 kHz VAE |
| `v1-44.pth` | 44.1 kHz VAE |
| `best_netG.pt` | BigVGAN vocoder |
| `synchformer_state_dict.pth` | synchronization features |

The default `small_16k` run uses `v1-16.pth`; 44.1 kHz variants use
`v1-44.pth`. You can change the VAE, vocoder, and Synchformer paths in
`configs/base_config.yaml`. Pass the backbone initialization checkpoint
separately to training as `weights=/path/to/mmaudio_small_16k.pth`.
The `av-benchmark` downloader below does not populate `ext_weights/`.

PAVAS uses two visual encoders: `apple/DFN5B-CLIP-ViT-H-14-384` for the
MMAudio visual features and `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` for
offline object patch features. Their weights are downloaded automatically on
first use and reused from the Hugging Face cache afterward. For offline runs,
populate that cache before patch extraction, training, or evaluation.

For cache-based evaluation and precomputed-result metrics, download the public
`av-benchmark` weights after installation:

```bash
python scripts/download_eval_assets.py
```

Reference:

- https://github.com/hkchengrex/MMAudio/blob/main/docs/MODELS.md

## Step 6. Prepare The Offline Precompute Stages

PAVAS expects three offline assets before training:

1. VGGSound feature caches
2. CLIP patch features
3. PPE outputs

The recommended order is:

1. prepare VGGSound feature caches
2. prepare CLIP patch features
3. prepare PPE outputs

The detailed guides are:

- [precompute/mmaudio_cache/README.md](precompute/mmaudio_cache/README.md)
- [precompute/clip_patches/README.md](precompute/clip_patches/README.md)
- [precompute/ppe/README.md](precompute/ppe/README.md)

### Step 6.1. Prepare VGGSound Feature Caches

Follow the upstream MMAudio feature-cache pipeline:

- https://github.com/hkchengrex/MMAudio/blob/main/docs/TRAINING.md

### Step 6.2. Prepare CLIP Patch Features

After activating `.venv`, run one of the patch extraction scripts documented in:

- [precompute/clip_patches/README.md](precompute/clip_patches/README.md)

### Step 6.3. Prepare PPE Outputs

PPE is already vendored in this repository under `precompute/ppe/`. You do not need a separate checkout of Qwen2.5-VL, Grounded-SAM-2, or CUT3R for the default path.

#### 6.3.1 Download PPE Assets

```bash
source .venv/bin/activate
python precompute/ppe/scripts/download_ppe_assets.py
```

This downloads or prefetches:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `microsoft/Florence-2-large`
- `sam2.1_hiera_large.pt`
- `cut3r_512_dpt_4_64.pth`

#### 6.3.2 Set The VGGSound Video Root

```bash
export PAVAS_VGGSOUND_VIDEO_DIR=/path/to/vggsound/video
```

Optional checkpoint overrides are also supported:

```bash
export PAVAS_GSAM2_CKPT=/path/to/sam2.1_hiera_large.pt
export PAVAS_CUT3R_MODEL_PATH=/path/to/cut3r_512_dpt_4_64.pth
```

#### 6.3.3 Run PPE In Order

Run the all-stage pipeline once per split (train, val, and test):

```bash
python precompute/ppe/scripts/run_ppe_pipeline.py \
  --tsv sets/vgg-train.tsv \
  --video-root /path/to/vggsound/video \
  --output-root /path/to/ppe
```

The pipeline runs the wrappers in this order:

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

It extracts the required 8 fps frames and runs velocity for every discovered
prompt automatically. Use the same command with the val and test TSVs; see
[precompute/ppe/README.md](precompute/ppe/README.md) for sharding, retries,
and individual wrapper commands.

## Step 7. Fill In Your Local Config Paths

Before training or evaluation, review these files in order:

1. `configs/data/base.yaml` for feature-cache, CLIP patch, PPE, and GT cache paths
2. `configs/base_config.yaml` for external checkpoint paths
3. `configs/train/finetune.yaml` for training settings
4. `configs/eval/eval.yaml` for evaluation settings

`configs/paths.example.yaml` is only a worksheet; editing it does not change
the active Hydra configuration. Each CLIP patch `tsv` must be the index
generated alongside the configured `npy` for the same split.

Make sure they point to your:

- dataset roots
- VGGSound feature caches
- CLIP patch features
- PPE outputs
- external weights
- checkpoints
- output directories

## Step 8. Launch Training Or Evaluation

Once the environment, dataset, weights, and precompute outputs are ready, continue with:

- [docs/training.md](docs/training.md)
- [docs/evaluation.md](docs/evaluation.md)
