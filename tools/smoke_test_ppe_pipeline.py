"""Exercise PPE orchestration, output contracts, and resumability without models."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from precompute.ppe.scripts import run_ppe_pipeline as pipeline


with tempfile.TemporaryDirectory(prefix="pavas-ppe-pipeline-") as tmp:
    root = Path(tmp)
    videos = root / "videos"
    videos.mkdir()
    for video_id in ("-moving", "stationary"):
        (videos / f"{video_id}.mp4").write_bytes(b"video")
    tsv = root / "videos.tsv"
    tsv.write_text("id\tlabel\n-moving\tsound\nstationary\tsilence\n", encoding="utf-8")
    shard_0 = pipeline.parse_videos(tsv, videos, 0, 2, 0)
    shard_1 = pipeline.parse_videos(tsv, videos, 1, 2, 0)
    assert {video_id for video_id, _ in shard_0}.isdisjoint(
        video_id for video_id, _ in shard_1)
    assert {video_id for video_id, _ in shard_0 + shard_1} == {"-moving", "stationary"}
    output = root / "ppe"
    args = ["--tsv", str(tsv), "--video-root", str(videos), "--output-root", str(output)]
    calls = []

    def fake_frames(video, frame_dir, marker, ffmpeg, duration=8.0, frame_count=None):
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            (frame_dir / f"{index:06d}.jpg").write_bytes(b"jpg")
        marker.parent.mkdir(parents=True, exist_ok=True)
        stat = video.stat()
        marker.write_text(json.dumps({"video": str(video), "size": stat.st_size,
                                      "mtime_ns": stat.st_mtime_ns, "fps": 8,
                                      "duration": duration, "count": 2}))

    def fake_run(cmd, env):
        name = Path(cmd[1]).name
        calls.append(name)
        if name == "run_moving_object_discovery.py":
            list_file = Path(cmd[cmd.index("--list-file") + 1])
            moving_root = Path(cmd[cmd.index("--out-root-base") + 1])
            for line in list_file.read_text().splitlines():
                video = Path(line)
                target = moving_root / video.stem / "physics_moving.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                objects = {"object": "moving"} if video.stem == "-moving" else {}
                target.write_text(json.dumps({"video": str(video), "map": objects}))
        elif name == "run_mass_estimation.py":
            mass_root = Path(cmd[cmd.index("--out-root-base") + 1])
            target = mass_root / "-moving" / "mass_estimation.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({
                "video": str(videos / "-moving.mp4"),
                "map": {"object": "moving"},
                "mass_estimation": {"object": {"weight_kg": 1.0}},
            }))
        elif name == "run_segmentation.py":
            seg_root = Path(cmd[cmd.index("--output-root") + 1]) / "-moving" / "0"
            mask_dir = seg_root / "FL2_mask_data"
            prompt_dir = seg_root / "FL2_text_prompt"
            mask_dir.mkdir(parents=True, exist_ok=True)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / "prompt.json").write_text('{"prompt": "object"}')
            for index in range(2):
                np.savez_compressed(mask_dir / f"mask_{index:06d}.npz", mask=np.ones((2, 2)))
        elif name == "run_velocity_estimation.py":
            target = Path(cmd[cmd.index("--output-dir") + 1]) / "0" / "speeds" / "instant_speeds.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "video_id,prompt_id,text_prompt,label_id,from_frame,to_frame,speed_mps,present_from,present_to\n"
                "-moving,0,object,1,0,1,0.5,True,True\n")
        elif name == "verify_ppe_outputs.py":
            subprocess.run(cmd, env=env, check=True, capture_output=True)
        else:
            raise AssertionError(name)

    with patch.object(pipeline, "run", side_effect=fake_run), \
         patch.object(pipeline, "extract_frames", side_effect=fake_frames), \
         patch.object(pipeline, "ffmpeg_binary", return_value="ffmpeg"):
        assert pipeline.main(args) == 0
        summary = json.loads((output / "work" / "videos" / "shard_0000" / "summary.json").read_text())
        assert summary["videos"]["-moving"]["status"] == "complete"
        assert summary["videos"]["stationary"]["status"] == "no_moving_objects"
        assert (output / "velocity" / "stationary" / "no_moving_objects.json").is_file()
        assert calls.count("run_velocity_estimation.py") == 1

        calls.clear()
        assert pipeline.main(args) == 0
        assert calls == ["verify_ppe_outputs.py"]

        stale_mask = output / "segmentation" / "-moving" / "0" / "FL2_mask_data" / "mask_999999.npz"
        np.savez_compressed(stale_mask, mask=np.ones((2, 2)))
        calls.clear()
        variable_args = args + ["--duration", "0.25", "--frame-count", "2"]
        assert pipeline.main(variable_args) == 0
        assert calls == ["run_segmentation.py", "run_velocity_estimation.py", "verify_ppe_outputs.py"]
        assert not stale_mask.exists()

        (videos / "-moving.mp4").write_bytes(b"changed-video")
        calls.clear()
        assert pipeline.main(variable_args) == 0
        assert calls == [
            "run_moving_object_discovery.py", "run_mass_estimation.py",
            "run_segmentation.py", "run_velocity_estimation.py", "verify_ppe_outputs.py",
        ]
        args = variable_args

        calls.clear()
        assert pipeline.main(args + ["--force"]) == 0
        assert calls == [
            "run_moving_object_discovery.py", "run_mass_estimation.py",
            "run_segmentation.py", "run_velocity_estimation.py", "verify_ppe_outputs.py",
        ]

        speed = output / "velocity" / "-moving" / "0" / "speeds" / "instant_speeds.csv"
        assert pipeline.velocity_ready(speed, "-moving", 0, 2)
        assert not pipeline.velocity_ready(speed, "-moving", 0, 3)

        speed.write_text("broken\n")
        calls.clear()
        assert pipeline.main(args) == 0
        assert calls == ["run_velocity_estimation.py", "verify_ppe_outputs.py"]

        mass = output / "mass" / "-moving" / "mass_estimation.json"
        mass.write_text(json.dumps({
            "video": str(videos / "-moving.mp4"),
            "map": {"object": "moving"},
            "mass_estimation": {"object": {}},
        }))
        calls.clear()
        assert pipeline.main(args) == 0
        assert calls == ["run_mass_estimation.py", "verify_ppe_outputs.py"]

        speed.write_text(
            "video_id,prompt_id,text_prompt,label_id,from_frame,to_frame,speed_mps,present_from,present_to\n"
            "-moving,0,object,1,0,1,0.5,True,True\n"
            "-moving,0,object,1,1,2,nan,True,True\n"
        )
        calls.clear()
        assert pipeline.main(args) == 0
        assert calls == ["run_velocity_estimation.py", "verify_ppe_outputs.py"]

        missing_tsv = root / "missing.tsv"
        missing_tsv.write_text("id\tlabel\nmissing\tsound\n", encoding="utf-8")
        calls.clear()
        assert pipeline.main(["--tsv", str(missing_tsv), "--video-root", str(videos),
                              "--output-root", str(output)]) == 1
        assert calls == []
        missing_summary = json.loads((output / "work" / "missing" / "shard_0000" / "summary.json").read_text())
        assert missing_summary["videos"]["missing"]["status"] == "failed"

print("smoke_test_ppe_pipeline: ok")
