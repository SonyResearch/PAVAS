#!/usr/bin/env python3
"""Run the vendored PPE stages for one deterministic shard of a video TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def parse_videos(tsv: Path, video_root: Path, shard_index: int, num_shards: int, limit: int) -> list[tuple[str, Path]]:
    videos = []
    seen = set()
    with tsv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"TSV must contain an id column: {tsv}")
        for row in reader:
            video_id = (row.get("id") or "").strip()
            if not video_id or Path(video_id).name != video_id or video_id in {".", ".."}:
                raise ValueError(f"Invalid video id: {video_id!r}")
            if video_id in seen:
                raise ValueError(f"Duplicate video id: {video_id}")
            seen.add(video_id)
            shard = int.from_bytes(hashlib.blake2b(video_id.encode(), digest_size=8).digest(), "big") % num_shards
            if shard == shard_index:
                videos.append((video_id, video_root / f"{video_id}.mp4"))
                if limit and len(videos) >= limit:
                    break
    return videos


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def moving_map(path: Path, video: Path) -> dict | None:
    data = read_json(path)
    if data is None or not isinstance(data.get("map"), dict):
        return None
    recorded_video = data.get("video")
    if not isinstance(recorded_video, str) or Path(recorded_video).resolve() != video.resolve():
        return None
    return data["map"]


def mass_ready(path: Path, objects: dict, video: Path) -> bool:
    data = read_json(path)
    estimates = data.get("mass_estimation") if data else None
    recorded_video = data.get("video") if data else None
    if (not isinstance(recorded_video, str)
            or Path(recorded_video).resolve() != video.resolve()
            or data.get("map") != objects
            or not isinstance(estimates, dict)
            or not set(objects).issubset(estimates)):
        return False
    for name in objects:
        item = estimates[name]
        if not isinstance(item, dict):
            return False
        if "weight_kg" not in item:
            return False
        weight = item["weight_kg"]
        if weight is not None and (
            not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0
        ):
            return False
    return True


def source_ready(marker: Path, video: Path) -> bool:
    data = read_json(marker)
    if data is None or not video.exists():
        return False
    stat = video.stat()
    return (data.get("video") == str(video)
            and data.get("size") == stat.st_size
            and data.get("mtime_ns") == stat.st_mtime_ns)


def write_source_marker(marker: Path, video: Path) -> None:
    stat = video.stat()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "video": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }), encoding="utf-8")


def write_no_moving_marker(velocity_root: Path, video_id: str, video: Path) -> None:
    marker = velocity_root / video_id / "no_moving_objects.json"
    write_source_marker(marker, video)


def frame_ready(frame_dir: Path, marker: Path, video: Path, duration: float = 8.0,
                frame_count: int | None = None) -> bool:
    data = read_json(marker)
    return (source_ready(marker, video) and data.get("fps") == 8
            and data.get("duration", 8.0) == duration
            and (frame_count is None or data.get("count") == frame_count)
            and data.get("count", 0) >= 2
            and len(list(frame_dir.glob("*.jpg"))) == data["count"])


def extract_frames(video: Path, frame_dir: Path, marker: Path, ffmpeg: str,
                   duration: float = 8.0, frame_count: int | None = None) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("*.jpg"):
        old_frame.unlink()
    marker.unlink(missing_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-t", str(duration), "-vf", "fps=8",
    ]
    if frame_count is not None:
        command.extend(["-frames:v", str(frame_count)])
    command.append(str(frame_dir / "%06d.jpg"))
    subprocess.run(command, check=True)
    count = len(list(frame_dir.glob("*.jpg")))
    if count < 2:
        raise RuntimeError(f"Frame extraction yielded only {count} frames: {video}")
    if frame_count is not None and count != frame_count:
        raise RuntimeError(f"Expected {frame_count} frames, got {count}: {video}")
    stat = video.stat()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "video": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "fps": 8, "duration": duration, "count": count,
    }), encoding="utf-8")


def segmentation_ready(seg_root: Path, video_id: str, objects: dict, frame_count: int) -> bool:
    for prompt_id, name in enumerate(objects):
        prompt_dir = seg_root / video_id / str(prompt_id)
        prompt = read_json(prompt_dir / "FL2_text_prompt" / "prompt.json")
        masks = list((prompt_dir / "FL2_mask_data").glob("mask_*.npz"))
        if prompt is None or prompt.get("prompt") != name or len(masks) != frame_count:
            return False
    return True


def velocity_ready(path: Path, video_id: str, prompt_id: int, frame_count: int) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"video_id", "prompt_id", "label_id", "speed_mps",
                        "from_frame", "to_frame"}
            if not required.issubset(reader.fieldnames or []):
                return False
            rows = list(reader)
            timelines: dict[int, set[int]] = {}
            for row in rows:
                if row["video_id"] != video_id or row["prompt_id"] != str(prompt_id):
                    return False
                label_id = int(row["label_id"])
                from_frame = int(row["from_frame"])
                to_frame = int(row["to_frame"])
                if from_frame != to_frame - 1 or not 1 <= to_frame < frame_count:
                    return False
                timelines.setdefault(label_id, set()).add(to_frame)
                if row["speed_mps"] not in {"", "None"}:
                    speed = float(row["speed_mps"])
                    if not math.isfinite(speed) or speed < 0:
                        return False
            expected = set(range(1, frame_count))
            return bool(timelines) and all(frames == expected for frames in timelines.values())
    except (OSError, TypeError, ValueError):
        return False


def selected_moving_dir(parent: Path, ids: list[str], moving_root: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for video_id in ids:
        target = parent / video_id
        target.mkdir()
        (target / "physics_moving.json").symlink_to(
            (moving_root / video_id / "physics_moving.json").resolve())
    return parent


def clear_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def ffmpeg_binary(value: str | None) -> str:
    if value:
        path = shutil.which(value)
        if path is None:
            raise RuntimeError(f"FFmpeg executable not found: {value}")
        return path
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("FFmpeg not found. Install imageio-ffmpeg or pass --ffmpeg-bin.") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True, help="Input TSV with an id column, e.g. sets/vgg-train.tsv")
    parser.add_argument("--video-root", type=Path, required=True, help="Directory of {id}.mp4 videos")
    parser.add_argument("--output-root", type=Path, required=True, help="PPE output root; configure its mass/velocity/segmentation subdirs in PAVAS")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Seconds to process for frame-based stages (default: 8)")
    parser.add_argument("--frame-count", type=int, default=None,
                        help="Require exactly this many 8-fps frames")
    parser.add_argument("--discovery-timeout", type=int, default=12,
                        help="Per-video Qwen discovery timeout in seconds (default: 12)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N videos in this shard (for testing)")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--seg-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Recompute existing PPE outputs")
    parser.add_argument("--dry-run", action="store_true", help="List selected videos without loading any models")
    args = parser.parse_args(argv)

    if (args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards
            or args.limit < 0 or args.seg_workers < 1
            or not math.isfinite(args.duration) or args.duration <= 0
            or (args.frame_count is not None and args.frame_count < 2)
            or args.discovery_timeout < 1):
        parser.error("Invalid shard, limit, or worker arguments")
    tsv = args.tsv.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    videos = parse_videos(tsv, video_root, args.shard_index, args.num_shards, args.limit)
    print(f"Selected {len(videos)} videos from {tsv} (shard {args.shard_index}/{args.num_shards})", flush=True)
    if args.dry_run:
        for video_id, video in videos:
            print(f"{video_id}\t{video}")
        return 0
    if not videos:
        return 0

    work = output_root / "work" / tsv.stem / f"shard_{args.shard_index:04d}"
    moving_root = work / "moving"
    mass_root = output_root / "mass"
    seg_root = output_root / "segmentation"
    velocity_root = output_root / "velocity"
    frame_root = work / "frames"
    for directory in (work, moving_root, mass_root, seg_root, velocity_root, frame_root):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PAVAS_VGGSOUND_VIDEO_DIR"] = str(video_root)
    results = {video_id: {"status": "pending", "prompts": 0, "errors": []} for video_id, _ in videos}
    available = [(video_id, video) for video_id, video in videos if video.is_file()]
    for video_id, video in videos:
        if not video.is_file():
            results[video_id]["errors"].append(f"Video not found: {video}")

    source_changed = {
        video_id for video_id, video in available
        if args.force or not source_ready(work / "source_done" / f"{video_id}.json", video)
    }
    pending = [
        (video_id, video) for video_id, video in available
        if video_id in source_changed
        or moving_map(moving_root / video_id / "physics_moving.json", video) is None
    ]
    moving_refreshed = {video_id for video_id, _ in pending}
    if pending:
        for video_id, _ in pending:
            clear_tree(moving_root / video_id)
            clear_tree(mass_root / video_id)
            clear_tree(seg_root / video_id)
            clear_tree(velocity_root / video_id)
        list_file = work / "moving_pending.txt"
        list_file.write_text("".join(f"{video}\n" for _, video in pending), encoding="utf-8")
        run([sys.executable, str(SCRIPTS / "run_moving_object_discovery.py"),
             "--list-file", str(list_file), "--out-root-base", str(moving_root),
             "--out-subdir", ".", "--timeout", str(args.discovery_timeout)], env)

    objects_by_id = {}
    for video_id, video in available:
        objects = moving_map(moving_root / video_id / "physics_moving.json", video)
        if objects is None:
            results[video_id]["errors"].append("Moving-object output missing or invalid")
        elif not objects:
            write_source_marker(work / "source_done" / f"{video_id}.json", video)
            write_no_moving_marker(velocity_root, video_id, video)
            results[video_id]["status"] = "no_moving_objects"
        else:
            write_source_marker(work / "source_done" / f"{video_id}.json", video)
            (velocity_root / video_id / "no_moving_objects.json").unlink(missing_ok=True)
            objects_by_id[video_id] = objects
            results[video_id]["prompts"] = len(objects)

    pending_ids = [
        video_id for video_id, objects in objects_by_id.items()
        if video_id in moving_refreshed
        or not mass_ready(
            mass_root / video_id / "mass_estimation.json", objects,
            dict(available)[video_id],
        )
    ]
    if pending_ids:
        with tempfile.TemporaryDirectory(prefix="mass-pending-", dir=work) as tmp:
            selected = selected_moving_dir(Path(tmp), pending_ids, moving_root)
            cmd = [sys.executable, str(SCRIPTS / "run_mass_estimation.py"),
                   "--in-root-base", str(selected), "--in-subdir", ".",
                   "--out-root-base", str(mass_root), "--out-subdir", "."]
            cmd.append("--overwrite")
            run(cmd, env)

    ready = {}
    frames_refreshed = set()
    for video_id, video in available:
        objects = objects_by_id.get(video_id)
        if objects is None:
            continue
        if not mass_ready(mass_root / video_id / "mass_estimation.json", objects, video):
            results[video_id]["errors"].append("Mass output missing or invalid")
            continue
        frame_dir = frame_root / video_id
        marker = work / "frame_done" / f"{video_id}.json"
        if args.force or not frame_ready(frame_dir, marker, video, args.duration, args.frame_count):
            try:
                extract_frames(video, frame_dir, marker, ffmpeg_binary(args.ffmpeg_bin),
                               args.duration, args.frame_count)
                frames_refreshed.add(video_id)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                results[video_id]["errors"].append(f"Frame extraction failed: {exc}")
                continue
        ready[video_id] = (objects, frame_dir, len(list(frame_dir.glob("*.jpg"))))

    pending_ids = [
        video_id for video_id, (objects, _, count) in ready.items()
        if args.force or video_id in moving_refreshed or video_id in frames_refreshed
        or not segmentation_ready(seg_root, video_id, objects, count)
    ]
    segmentation_refreshed = set(pending_ids)
    if pending_ids:
        for video_id in pending_ids:
            clear_tree(seg_root / video_id)
            clear_tree(velocity_root / video_id)
        with tempfile.TemporaryDirectory(prefix="seg-pending-", dir=work) as tmp:
            selected = selected_moving_dir(Path(tmp), pending_ids, moving_root)
            run([sys.executable, str(SCRIPTS / "run_segmentation.py"),
                 "--moving-root", str(selected), "--manifest-out", str(work / "manifests"),
                 "--frames-root", str(frame_root), "--output-root", str(seg_root),
                 "--num-workers", str(args.seg_workers), "--no-box-viz", "--no-video"], env)

    for video_id, (objects, frame_dir, count) in ready.items():
        if not segmentation_ready(seg_root, video_id, objects, count):
            results[video_id]["errors"].append("Segmentation output missing or incomplete")
            continue
        for prompt_id in range(len(objects)):
            prompt_dir = seg_root / video_id / str(prompt_id)
            csv_path = velocity_root / video_id / str(prompt_id) / "speeds" / "instant_speeds.csv"
            if (args.force or video_id in segmentation_refreshed
                    or not velocity_ready(csv_path, video_id, prompt_id, count)):
                try:
                    run([sys.executable, str(SCRIPTS / "run_velocity_estimation.py"),
                         "--seq-path", str(frame_dir),
                         "--mask-path", str(prompt_dir / "FL2_mask_data"),
                         "--prompt-path", str(prompt_dir / "FL2_text_prompt"),
                         "--output-dir", str(velocity_root / video_id),
                         "--prompt-id", str(prompt_id)], env)
                except subprocess.CalledProcessError as exc:
                    results[video_id]["errors"].append(f"Velocity prompt {prompt_id} failed: {exc}")
                    continue
            if not velocity_ready(csv_path, video_id, prompt_id, count):
                results[video_id]["errors"].append(
                    f"Velocity prompt {prompt_id} output is missing or incomplete"
                )
                continue
            try:
                run([sys.executable, str(SCRIPTS / "verify_ppe_outputs.py"),
                     f"--video-id={video_id}", "--prompt-id", str(prompt_id),
                     "--mass-root", str(mass_root), "--velocity-root", str(velocity_root),
                     "--seg-root", str(seg_root)], env)
            except subprocess.CalledProcessError as exc:
                results[video_id]["errors"].append(f"Validation prompt {prompt_id} failed: {exc}")

    for item in results.values():
        if item["errors"]:
            item["status"] = "failed"
        elif item["status"] == "pending":
            item["status"] = "complete"
    summary = {"tsv": str(tsv), "shard_index": args.shard_index,
               "num_shards": args.num_shards, "videos": results}
    summary_path = work / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    counts = {status: sum(item["status"] == status for item in results.values())
              for status in ("complete", "no_moving_objects", "failed")}
    print(f"PPE summary: {counts}; details: {summary_path}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
