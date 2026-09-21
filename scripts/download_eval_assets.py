#!/usr/bin/env python3
"""Download the public checkpoints required by av-benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import urllib.request
from pathlib import Path


ASSETS = {
    "music_speech_audioset_epoch_15_esc_89.98.pt": (
        "https://huggingface.co/lukewys/laion_clap/resolve/main/"
        "music_speech_audioset_epoch_15_esc_89.98.pt"
    ),
    "synchformer_state_dict.pth": (
        "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
        "synchformer_state_dict.pth"
    ),
}


def benchmark_weights_dir() -> Path:
    spec = importlib.util.find_spec("av_bench")
    if spec is None or spec.origin is None:
        raise SystemExit("Install av-benchmark first with scripts/setup_env.sh")
    return Path(spec.origin).resolve().parent.parent / "weights"


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 1_000_000:
        print(f"[skip] {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url) as response:
                while chunk := response.read(1024 * 1024):
                    tmp.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    if temp_path.stat().st_size <= 1_000_000:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {url}")
    os.replace(temp_path, destination)
    print(f"[done] {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the av-benchmark weights directory.")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser() if args.output_dir else benchmark_weights_dir()
    for name, url in ASSETS.items():
        download(url, output_dir / name)


if __name__ == "__main__":
    main()
