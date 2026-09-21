#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
CUT3R_FILE_ID = "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD"
CUT3R_DRIVE_URL = f"https://drive.google.com/uc?id={CUT3R_FILE_ID}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1_000_000


def download_http(url: str, destination: Path) -> None:
    ensure_parent(destination)
    if ready(destination):
        print(f"[skip] exists: {destination}")
        return
    print(f"[download] {url}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, temp_path)
        if not ready(temp_path):
            raise RuntimeError(f"Downloaded file is too small: {url}")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"[done] {destination}")


def download_cut3r(destination: Path) -> None:
    ensure_parent(destination)
    if ready(destination):
        print(f"[skip] exists: {destination}")
        return
    print(f"[download] CUT3R checkpoint -> {destination}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        try:
            result = gdown.download(CUT3R_DRIVE_URL, str(temp_path), quiet=False)
        except TypeError:
            result = gdown.download(CUT3R_DRIVE_URL, str(temp_path))
        if result is None or not ready(temp_path):
            raise RuntimeError("CUT3R checkpoint download failed")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"[done] {destination}")


def prefetch_hf_model(model_id: str, cache_dir: Path | None) -> None:
    print(f"[prefetch] {model_id}")
    snapshot_download(repo_id=model_id, cache_dir=str(cache_dir) if cache_dir else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or prefetch PPE assets used by PAVAS.")
    parser.add_argument("--hf-cache-dir", type=Path, default=None,
                        help="Optional Hugging Face cache override. Set HF_HOME to the same directory before running PPE.")
    parser.add_argument("--sam2-dst", type=Path, default=REPO_ROOT / "grounded_sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    parser.add_argument("--cut3r-dst", type=Path, default=REPO_ROOT / "cut3r" / "src" / "cut3r_512_dpt_4_64.pth")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-florence", action="store_true")
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--skip-cut3r", action="store_true")
    args = parser.parse_args()

    hf_cache_dir = args.hf_cache_dir.expanduser().resolve() if args.hf_cache_dir else None
    if hf_cache_dir is not None:
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_qwen:
        prefetch_hf_model(QWEN_MODEL_ID, hf_cache_dir)
    if not args.skip_florence:
        prefetch_hf_model(FLORENCE_MODEL_ID, hf_cache_dir)
    if not args.skip_sam2:
        download_http(SAM2_URL, args.sam2_dst.expanduser().resolve())
    if not args.skip_cut3r:
        download_cut3r(args.cut3r_dst.expanduser().resolve())


if __name__ == "__main__":
    main()
