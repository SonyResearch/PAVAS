#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1] / "grounded_sam2"


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_base_command(
    *,
    python_bin: str,
    script_path: Path,
    singularity_image: str | None,
    singularity_bin: str,
    binds: list[str],
) -> list[str]:
    if singularity_image:
        cmd = [singularity_bin, "exec", "--nv"]
        for bind in binds:
            cmd.extend(["-B", bind])
        cmd.extend([singularity_image, python_bin, str(script_path)])
        return cmd
    return [python_bin, str(script_path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Florence2 + SAM2 segmentation for one moving-output directory."
    )
    parser.add_argument("--repo-root", type=Path, default=os.environ.get("PAVAS_GSAM2_REPO") or DEFAULT_REPO_ROOT)
    parser.add_argument("--python-bin", default=os.environ.get("PAVAS_GSAM2_PYTHON", "python"))
    parser.add_argument("--singularity-image", default=os.environ.get("PAVAS_GSAM2_SIF"))
    parser.add_argument("--singularity-bin", default=os.environ.get("PAVAS_SINGULARITY_BIN", "singularity"))
    parser.add_argument("--sam2-checkpoint", default=os.environ.get("PAVAS_GSAM2_CKPT", "checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument(
        "--moving-root",
        required=True,
        help="Directory containing physics_moving.json files, either directly or under one moving-output subdirectory.",
    )
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Index used only to name the manifest/output shard for this run.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--script-name", default="Florence2_train_rev.py")
    parser.add_argument("--no-box-viz", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if not args.repo_root:
        raise SystemExit("--repo-root or PAVAS_GSAM2_REPO is required")

    repo_root = Path(args.repo_root).expanduser().resolve()
    prep_script = repo_root / "prep_only_manifest.py"
    infer_script = repo_root / args.script_name
    if not prep_script.exists():
        raise SystemExit(f"Manifest prep script not found: {prep_script}")
    if not infer_script.exists():
        raise SystemExit(f"Segmentation script not found: {infer_script}")

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("KEEP_STDERR", "1")
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    sam2_ckpt = Path(args.sam2_checkpoint)
    if not sam2_ckpt.is_absolute():
        sam2_ckpt = repo_root / sam2_ckpt
    env["PAVAS_GSAM2_CKPT"] = str(sam2_ckpt)

    split_index = 0 if args.split_index is None else args.split_index
    prep_cmd = build_base_command(
        python_bin=args.python_bin,
        script_path=prep_script,
        singularity_image=args.singularity_image,
        singularity_bin=args.singularity_bin,
        binds=args.bind,
    )
    prep_cmd.extend([
        "--moving_root", args.moving_root,
        "--manifest_out", args.manifest_out,
        "--split_index", str(split_index),
    ])
    run(prep_cmd, repo_root, env)

    manifest_path = Path(args.manifest_out) / f"manifest_split_{split_index}.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        total = len(json.load(f))

    infer_cmd = build_base_command(
        python_bin=args.python_bin,
        script_path=infer_script,
        singularity_image=args.singularity_image,
        singularity_bin=args.singularity_bin,
        binds=args.bind,
    )
    output_dir = Path(args.output_root)
    if args.split_index is not None:
        output_dir /= f"split_{split_index:02d}"
    infer_cmd.extend([
        "--text", str(manifest_path),
        "--image_root", args.frames_root,
        "--output", str(output_dir),
        "--fps", str(args.fps),
        "--start_idx", "0",
        "--end_idx", str(total),
        "--num_workers", str(args.num_workers),
        "--run_tag", f"split_{split_index:02d}",
    ])
    if args.no_box_viz:
        infer_cmd.append("--no_box_viz")
    if args.no_video:
        infer_cmd.append("--no_video")
    run(infer_cmd, repo_root, env)


if __name__ == "__main__":
    main()
