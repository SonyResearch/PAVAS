#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_SPEED_COLUMNS = {
    "video_id",
    "prompt_id",
    "text_prompt",
    "label_id",
    "from_frame",
    "to_frame",
    "speed_mps",
    "present_from",
    "present_to",
}


def find_output(root: Path, relative_path: Path, pattern: str) -> Path:
    direct = root / relative_path
    if direct.exists():
        return direct
    matches = sorted(path for path in root.glob(pattern) if path.exists())
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one output for {relative_path} under {root}; found {len(matches)}. "
            "Pass the specific shard directory as the root if needed."
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PPE outputs for one video/prompt.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--mass-root", type=Path, required=True)
    parser.add_argument("--velocity-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    args = parser.parse_args()

    mass_json = find_output(
        args.mass_root,
        Path(args.video_id) / "mass_estimation.json",
        f"*/{args.video_id}/mass_estimation.json",
    )
    vel_csv = args.velocity_root / args.video_id / str(args.prompt_id) / "speeds" / "instant_speeds.csv"
    seg_dir = find_output(
        args.seg_root,
        Path(args.video_id) / str(args.prompt_id),
        f"split_*/{args.video_id}/{args.prompt_id}",
    )
    prompt_json = seg_dir / "FL2_text_prompt" / "prompt.json"
    mask_dir = seg_dir / "FL2_mask_data"
    npz_files = sorted(mask_dir.glob("mask_*.npz"))

    if not vel_csv.exists():
        raise SystemExit(f"Missing velocity CSV: {vel_csv}")
    if not prompt_json.exists():
        raise SystemExit(f"Missing prompt JSON: {prompt_json}")
    if not npz_files:
        raise SystemExit(f"No segmentation masks found under: {mask_dir}")

    with mass_json.open("r", encoding="utf-8") as f:
        mass_data = json.load(f)
    estimates = mass_data.get("mass_estimation")
    if not isinstance(estimates, dict) or not estimates:
        raise SystemExit(f"Unexpected mass JSON format: {mass_json}")
    for name, estimate in estimates.items():
        if not isinstance(estimate, dict) or "weight_kg" not in estimate:
            raise SystemExit(f"Missing weight_kg for {name}: {mass_json}")
        weight = estimate["weight_kg"]
        if weight is not None and (
            not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0
        ):
            raise SystemExit(f"Invalid weight_kg for {name}: {mass_json}")

    with prompt_json.open("r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    if "prompt" not in prompt_data:
        raise SystemExit(f"Prompt JSON missing 'prompt' key: {prompt_json}")

    with np.load(npz_files[0]) as mask_data:
        if "mask" not in mask_data or mask_data["mask"].ndim != 2:
            raise SystemExit(f"Invalid segmentation mask: {npz_files[0]}")

    with vel_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = REQUIRED_SPEED_COLUMNS - header
        if missing:
            raise SystemExit(f"Velocity CSV missing columns {sorted(missing)}: {vel_csv}")
        rows = list(reader)
        if not rows:
            raise SystemExit(f"Velocity CSV has no rows: {vel_csv}")
        for row in rows:
            if row["video_id"] != args.video_id or row["prompt_id"] != str(args.prompt_id):
                raise SystemExit(f"Velocity CSV identifiers do not match: {vel_csv}")
            if row["speed_mps"] not in {"", "None"}:
                try:
                    speed = float(row["speed_mps"])
                    if not math.isfinite(speed) or speed < 0:
                        raise ValueError
                except ValueError:
                    raise SystemExit(f"Invalid speed_mps in {vel_csv}") from None

    print("verify_ppe_outputs: ok")
    print(f"mass_json={mass_json}")
    print(f"velocity_csv={vel_csv}")
    print(f"prompt_json={prompt_json}")
    print(f"mask_count={len(npz_files)}")


if __name__ == "__main__":
    main()
