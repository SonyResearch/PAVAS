import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "precompute" / "ppe" / "scripts" / "verify_ppe_outputs.py"
assert VALIDATOR.exists()
sys.path.insert(0, str(REPO_ROOT / "pavas"))

from core.data.object_conditions import build_vggsound_val_inputs
from core.data.extracted_vgg import ExtractedVGG
from core.data.mm_dataset import collate_fn
from core.model.networks import MMAudio


def check_layout(root: Path, sharded: bool) -> None:
    video_id = "example_video"
    prompt_id = "0"
    mass_root = root / "mass"
    seg_root = root / "segmentation"
    velocity_root = root / "velocity"
    mass_dir = mass_root / ("split_00_json" if sharded else "") / video_id
    seg_dir = seg_root / ("split_00" if sharded else "") / video_id / prompt_id
    vel_dir = velocity_root / video_id / prompt_id / "speeds"
    mask_dir = seg_dir / "FL2_mask_data"
    prompt_dir = seg_dir / "FL2_text_prompt"
    for path in (mass_dir, vel_dir, mask_dir, prompt_dir):
        path.mkdir(parents=True)

    (mass_dir / "mass_estimation.json").write_text(
        json.dumps({"mass_estimation": {"ball": {"weight_kg": 0.5}}}), encoding="utf-8"
    )
    (prompt_dir / "prompt.json").write_text(json.dumps({"prompt": "ball"}), encoding="utf-8")
    np.savez_compressed(mask_dir / "mask_000000.npz", mask=np.ones((4, 4), dtype=np.uint8))
    np.savez_compressed(mask_dir / "mask_000001.npz", mask=np.ones((4, 4), dtype=np.uint8))
    with (vel_dir / "instant_speeds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "prompt_id", "text_prompt", "label_id", "from_frame", "to_frame",
            "speed_mps", "present_from", "present_to",
        ])
        writer.writeheader()
        writer.writerow({
            "video_id": video_id, "prompt_id": prompt_id, "text_prompt": "ball",
            "label_id": 1, "from_frame": 0, "to_frame": 1, "speed_mps": 2.0,
            "present_from": 1, "present_to": 1,
        })

    result = subprocess.run([
        sys.executable, "-B", str(VALIDATOR), "--video-id", video_id,
        "--prompt-id", prompt_id, "--mass-root", str(mass_root),
        "--velocity-root", str(velocity_root), "--seg-root", str(seg_root),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    sample = build_vggsound_val_inputs(
        video_id, int(prompt_id), mass_root=mass_root, velocity_root=velocity_root,
        seg_root=seg_root, tv=2, patch_grid_hw=(2, 2), enable_patch_attn=True,
    )
    assert sample.mass_f.shape == (1, 1) and sample.mass_f[0, 0].item() == 0.5
    assert sample.velocity_f.shape == (1, 2, 1) and sample.velocity_f[0, 1, 0].item() == 2.0
    assert sample.obj_patch_attn is not None and sample.obj_patch_attn.shape == (1, 2, 4)

    if not sharded:
        dataset = ExtractedVGG.__new__(ExtractedVGG)
        dataset.df_list = [{"id": video_id, "label": "ball"}]
        dataset.mean = torch.zeros(1, 1, 1)
        dataset.std = torch.ones(1, 1, 1)
        dataset.clip_features = torch.zeros(1, 1, 1)
        dataset.sync_features = torch.zeros(1, 1, 1)
        dataset.text_features = torch.zeros(1, 1, 1)
        dataset.video_exist = torch.tensor(True)
        dataset.text_exist = torch.tensor(True)
        dataset.video_root = None
        dataset.clip_patches = np.ones((1, 2, 2, 2, 4), dtype=np.float32)
        dataset._clip_patches_id2idx = {video_id: 0}
        dataset._object_cond = {
            "enabled": True,
            "enable_patch_attn": True,
            "mass_root": str(mass_root),
            "velocity_root": str(velocity_root),
            "seg_root": str(seg_root),
        }
        dataset._split_tag = "test"
        test_sample = dataset[0]
        for key in ("clip_patches", "mass_f", "velocity_f", "obj_patch_attn", "presence_mask"):
            assert key in test_sample, f"test split is missing {key}"

        dataset._clip_patches_id2idx = {}
        with patch('core.data.extracted_vgg.log.warning') as warning:
            missing_patch = dataset[0]
        assert 'clip_patches' not in missing_patch
        assert 'mass_f' in missing_patch
        assert any('patch_id_missing' in str(call) for call in warning.call_args_list)

        with patch('core.data.extracted_vgg.build_vggsound_val_inputs_all_prompts') as build:
            build.return_value = SimpleNamespace(
                velocity_f=None, mass_f=None, presence=None, obj_patch_attn=None,
                label_ids=None, prompt_ids=None, velocity_missing=None, stage_prof=None,
            )
            with patch('core.data.extracted_vgg.log.warning') as warning:
                no_object = dataset[0]
        assert 'mass_f' not in no_object and 'velocity_f' not in no_object
        assert any('object_inputs_absent_or_no_moving' in str(call) for call in warning.call_args_list)

        with patch('core.data.mm_dataset.log.warning') as warning:
            mixed_batch = collate_fn([test_sample, missing_patch])
        assert 'clip_patches' not in mixed_batch
        assert any('mixed_patch_batch' in str(call) for call in warning.call_args_list)

        model_stub = SimpleNamespace(arch_debug=False, patch_to_hidden=SimpleNamespace(in_features=4))
        bad_condition = {
            'clip_patches': torch.zeros(1, 2, 2, 2, 4),
            'obj_patch_attn': torch.zeros(1, 1, 2),
        }
        with patch('core.model.networks.log.warning') as warning:
            framewise = MMAudio._build_framewise_object_features(model_stub, bad_condition)
        assert framewise == (None, None)
        assert any('framewise_object_feature_failed' in str(call) for call in warning.call_args_list)


with tempfile.TemporaryDirectory(prefix="pavas-ppe-smoke-") as tmp:
    check_layout(Path(tmp) / "direct", sharded=False)
    check_layout(Path(tmp) / "sharded", sharded=True)

print("smoke_test_ppe_inputs: ok")
