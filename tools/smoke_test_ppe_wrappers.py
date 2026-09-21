import json
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "precompute" / "ppe" / "scripts"


def run_wrapper(name: str, args: list[str]) -> None:
    with patch.object(sys, "argv", [str(SCRIPTS / name), *args]):
        runpy.run_path(str(SCRIPTS / name), run_name="__main__")


with tempfile.TemporaryDirectory(prefix="pavas-ppe-wrappers-") as tmp:
    root = Path(tmp)
    moving_root = root / "moving"
    mass_root = root / "mass"
    seg_root = root / "segmentation"
    manifest_root = root / "manifests"
    frames_root = root / "frames"
    (moving_root / "example").mkdir(parents=True)
    (moving_root / "example" / "physics_moving.json").write_text("{}", encoding="utf-8")

    with patch("subprocess.run") as process:
        run_wrapper("run_moving_object_discovery.py", ["--out-root-base", str(moving_root)])
        command = process.call_args.args[0]
        assert command[command.index("--out_subdir") + 1] == "."

    with patch("subprocess.run") as process:
        run_wrapper("run_mass_estimation.py", [
            "--in-root-base", str(moving_root), "--out-root-base", str(mass_root),
        ])
        command = process.call_args.args[0]
        assert command[command.index("--in_subdir") + 1] == "."

    def fake_segmentation(command: list[str], **kwargs) -> None:
        if "--manifest_out" in command:
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / "manifest_split_0.json").write_text(
                json.dumps({"example.mp4": {"final output": "{}"}}), encoding="utf-8"
            )

    with patch("subprocess.run", side_effect=fake_segmentation) as process:
        run_wrapper("run_segmentation.py", [
            "--moving-root", str(moving_root), "--manifest-out", str(manifest_root),
            "--frames-root", str(frames_root), "--output-root", str(seg_root),
        ])
        assert process.call_count == 2
        command = process.call_args.args[0]
        assert command[command.index("--output") + 1] == str(seg_root)

print("smoke_test_ppe_wrappers: ok")
