#!/usr/bin/env bash
set -euo pipefail

python3 -c 'import sys; sys.exit("Python 3.10+ is required") if sys.version_info < (3, 10) else None'
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
