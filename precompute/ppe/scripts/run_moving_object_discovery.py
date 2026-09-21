#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1] / "qwen2_5_vl"


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
    parser = argparse.ArgumentParser(description="Run Qwen moving-object discovery for a split/list.")
    parser.add_argument("--repo-root", type=Path, default=os.environ.get("PAVAS_QWEN_REPO") or DEFAULT_REPO_ROOT)
    parser.add_argument("--python-bin", default=os.environ.get("PAVAS_QWEN_PYTHON", "python"))
    parser.add_argument("--singularity-image", default=os.environ.get("PAVAS_QWEN_SIF"))
    parser.add_argument("--singularity-bin", default=os.environ.get("PAVAS_SINGULARITY_BIN", "singularity"))
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument("--list-file", default="")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--prestart-timeout", type=int, default=6)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--out-root-base", required=True)
    parser.add_argument("--out-subdir", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.repo_root:
        raise SystemExit("--repo-root or PAVAS_QWEN_REPO is required")

    repo_root = Path(args.repo_root).expanduser().resolve()
    script_path = repo_root / "qwen_vlm_bow_moving_slurm.py"
    if not script_path.exists():
        raise SystemExit(f"Moving-object script not found: {script_path}")

    cmd = build_base_command(
        python_bin=args.python_bin,
        script_path=script_path,
        singularity_image=args.singularity_image,
        singularity_bin=args.singularity_bin,
        binds=args.bind,
    )
    cmd.extend([
        "--pattern", args.pattern,
        "--limit", str(args.limit),
        "--timeout", str(args.timeout),
        "--prestart_timeout", str(args.prestart_timeout),
        "--device_map", args.device_map,
        "--out_root_base", args.out_root_base,
    ])
    if args.list_file:
        cmd.extend(["--list_file", args.list_file])
    cmd.extend(["--out_subdir", args.out_subdir or "."])
    if args.resume:
        cmd.append("--resume")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
