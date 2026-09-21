#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate segmentation manifests from moving-object JSONs without extracting frames.

Input:
  moving_root/split_x_json/<VID>/physics_moving.json
    {
      "video": ".../video/xxx.mp4",
      "map": {"object": "action", ...}
    }

Output:
  manifest_out/manifest_split_{IDX}.json
    {
      "xxx.mp4": {
        "final output": "{\"object\":\"action\", ...}"
      },
      ...
    }
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def find_moving_jsons(base: Path):
    return sorted([p for p in base.rglob("physics_moving.json") if p.is_file()])


def load_moving(path: Path):
    try:
        data = json.loads(path.read_text())
        video_path = data.get("video", "")
        objects_map = data.get("map", {})
        if not isinstance(objects_map, dict):
            objects_map = {}
        return video_path, objects_map
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return "", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moving_root", type=str, required=True,
                    help="Path to one split dir (e.g., split_0_json)")
    ap.add_argument("--manifest_out", type=str, required=True,
                    help="Output dir for manifest_split_XX.json")
    ap.add_argument("--split_index", type=int, required=True,
                    help="Split index (0..N-1)")
    args = ap.parse_args()

    moving_root = Path(args.moving_root)
    manifest_out = Path(args.manifest_out)
    manifest_out.mkdir(parents=True, exist_ok=True)

    items = find_moving_jsons(moving_root)
    print(f"[INFO] Found {len(items)} physics_moving.json under {moving_root}")

    manifest = {}
    for mp in items:
        video_path, obj_map = load_moving(mp)
        if not video_path or not obj_map:
            continue
        stem = Path(video_path).name  # keep full mp4 name
        # Save whole map as string
        manifest[stem] = {
            "final output": json.dumps(obj_map, ensure_ascii=False)
        }

    out_path = manifest_out / f"manifest_split_{args.split_index}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved manifest: {out_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
