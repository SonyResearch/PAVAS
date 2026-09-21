#!/usr/bin/env python3
"""Compute av-benchmark metrics from existing prediction and ground-truth caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--pred-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--skip-video-related", action="store_true")
    args = parser.parse_args()

    for root in (args.gt_cache, args.pred_cache):
        if not (root / "pann_features.pth").is_file():
            parser.error(f"Missing pann_features.pth in cache directory: {root}")

    from av_bench.evaluate import evaluate

    metrics = evaluate(
        gt_audio_cache=args.gt_cache,
        pred_audio_cache=args.pred_cache,
        skip_clap=args.skip_clap,
        skip_video_related=args.skip_video_related,
    )
    output = json.dumps({key: float(value) for key, value in metrics.items()}, indent=2)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
