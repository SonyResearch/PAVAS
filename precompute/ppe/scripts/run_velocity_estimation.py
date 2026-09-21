#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1] / "cut3r"


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
    parser = argparse.ArgumentParser(description="Run CUT3R-based velocity estimation for one video/prompt.")
    parser.add_argument("--repo-root", type=Path, default=os.environ.get("PAVAS_CUT3R_REPO") or DEFAULT_REPO_ROOT)
    parser.add_argument("--python-bin", default=os.environ.get("PAVAS_CUT3R_PYTHON", "python"))
    parser.add_argument("--singularity-image", default=os.environ.get("PAVAS_CUT3R_SIF"))
    parser.add_argument("--singularity-bin", default=os.environ.get("PAVAS_SINGULARITY_BIN", "singularity"))
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument("--seq-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-id", default="")
    parser.add_argument("--model-path", default=None, help="Optional CUT3R checkpoint override.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vis-threshold", type=float, default=0.7)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--save-summaries", action="store_true")
    parser.add_argument("--points-stride", type=int, default=1)
    args = parser.parse_args()

    if not args.repo_root:
        raise SystemExit("--repo-root or PAVAS_CUT3R_REPO is required")

    repo_root = Path(args.repo_root).expanduser().resolve()
    script_path = repo_root / "cut3r_vgg_instant_velocity.py"
    if not script_path.exists():
        raise SystemExit(f"Velocity script not found: {script_path}")

    model_path = args.model_path or os.environ.get("PAVAS_CUT3R_MODEL_PATH") or str(repo_root / "src" / "cut3r_512_dpt_4_64.pth")

    cmd = build_base_command(
        python_bin=args.python_bin,
        script_path=script_path,
        singularity_image=args.singularity_image,
        singularity_bin=args.singularity_bin,
        binds=args.bind,
    )
    cmd.extend([
        "--seq_path", args.seq_path,
        "--mask_path", args.mask_path,
        "--prompt_path", args.prompt_path,
        "--output_dir", args.output_dir,
        "--prompt_id", args.prompt_id,
        "--model_path", model_path,
        "--device", args.device,
        "--size", str(args.size),
        "--vis_threshold", str(args.vis_threshold),
        "--fps", str(args.fps),
        "--points_stride", str(args.points_stride),
    ])
    if args.save_summaries:
        cmd.append("--save_summaries")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
