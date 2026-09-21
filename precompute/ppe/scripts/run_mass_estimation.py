#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1] / "qwen2_5_vl"


def infer_input_subdir(root: Path) -> str:
    """Infer the moving-output subdirectory.

    Supported layouts:
    - root/<video_id>/physics_moving.json
    - root/<subdir>/<video_id>/physics_moving.json
    """
    if any(root.glob("*/physics_moving.json")):
        return "."

    candidates: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*/physics_moving.json")):
            candidates.append(child.name)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise SystemExit(
            f"Could not find physics_moving.json under {root}. "
            "Expected either root/<video_id>/physics_moving.json or "
            "root/<subdir>/<video_id>/physics_moving.json."
        )

    raise SystemExit(
        f"Multiple moving-output subdirectories found under {root}: {candidates}. "
        "Please specify one explicitly with --in-subdir."
    )


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
        description="Run Qwen-based mass estimation for one moving-output directory."
    )
    parser.add_argument("--repo-root", type=Path, default=os.environ.get("PAVAS_QWEN_REPO") or DEFAULT_REPO_ROOT)
    parser.add_argument("--python-bin", default=os.environ.get("PAVAS_QWEN_PYTHON", "python"))
    parser.add_argument("--singularity-image", default=os.environ.get("PAVAS_QWEN_SIF"))
    parser.add_argument("--singularity-bin", default=os.environ.get("PAVAS_SINGULARITY_BIN", "singularity"))
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument("--in-root-base", required=True)
    parser.add_argument(
        "--in-subdir",
        default="",
        help="Optional subdirectory under --in-root-base. If omitted, it is inferred automatically.",
    )
    parser.add_argument("--out-root-base", required=True)
    parser.add_argument("--out-subdir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.repo_root:
        raise SystemExit("--repo-root or PAVAS_QWEN_REPO is required")

    repo_root = Path(args.repo_root).expanduser().resolve()
    script_path = repo_root / "_qwen_mass_feature_moving_v2.py"
    if not script_path.exists():
        raise SystemExit(f"Mass script not found: {script_path}")

    cmd = build_base_command(
        python_bin=args.python_bin,
        script_path=script_path,
        singularity_image=args.singularity_image,
        singularity_bin=args.singularity_bin,
        binds=args.bind,
    )
    in_subdir = args.in_subdir.strip() or infer_input_subdir(Path(args.in_root_base))

    cmd.extend([
        "--in_root_base", args.in_root_base,
        "--in_subdir", in_subdir,
        "--out_root_base", args.out_root_base,
    ])
    if args.out_subdir:
        cmd.extend(["--out_subdir", args.out_subdir])
    if args.overwrite:
        cmd.append("--overwrite")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
