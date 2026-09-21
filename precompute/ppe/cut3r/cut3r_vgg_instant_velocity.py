#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUT3R-based 3D + segmentation-mask velocity estimation.

- Outputs are separated by prompt id:
  <output_dir>/<prompt_id>/{speeds,pts3ds_seg_* , color, camera_center, points3d}

- `points3d` is written to:
  <output_dir>/<prompt_id>/points3d/<instance_id>/<frame>.npy
  * Once an instance appears, a file is created for every frame.
  * Empty `(0, 3)` arrays are written when no points are available for that frame.
  * Frames before the first appearance are backfilled at the end of processing.

- Instantaneous speed CSV:
  <output_dir>/<prompt_id>/speeds/instant_speeds.csv
  * `dist_*` and `speed_*` fields are set to `"None"` when no centroid is available.
  * `present_from` and `present_to` record object visibility for each frame pair.
  * Every frame pair is emitted for each tracked label, including backfilled ranges.
"""

import os
import re
import cv2
import glob
import json
import time
import shutil
import random
import argparse
import tempfile
import numpy as np
import pandas as pd
import imageio.v2 as iio
import torch
import torch.nn.functional as F

from add_ckpt_path import add_path_to_dust3r
random.seed(42)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path", type=str, required=True, help="Frame directory or mp4 path.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--size", type=int, default=512, help="224 or 512 (model-dependent)")
    p.add_argument("--vis_threshold", type=float, default=0.7, help="conf > thr if conf available")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mask_path", type=str, required=True, help="Folder of .npz (key='mask')")
    p.add_argument("--prompt_path", type=str, required=True, help="Folder containing prompt.json")
    p.add_argument("--save_summaries", action="store_true", help="Save per-frame CSV/JSON/visual aids")
    p.add_argument("--fps", type=float, default=8.0, help="Frames per second. This should match the segmentation FPS.")
    p.add_argument("--save_instance_points", action="store_true", help="Save per-frame instance point clouds.")
    p.add_argument("--points_stride", type=int, default=1, help="Stride for point dumps to reduce disk usage.")
    p.add_argument("--prompt_id", type=str, default="", help="Prompt index. If omitted, infer it from the mask path.")
    return p.parse_args()


def _ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def parse_seq_path(p, target_fps=None):
    """Return image paths from a directory, or extract frames from an mp4."""
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(os.path.join(p, "*")))
        tmpdirname = None
        return img_paths, tmpdirname

    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise ValueError(f"Error opening video {p}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps == 0:
        cap.release()
        raise ValueError("Video FPS is 0")

    if target_fps is None or target_fps <= 0:
        target_fps = src_fps
    step = max(1, int(round(src_fps / target_fps)))
    frame_indices = list(range(0, total, step))

    img_paths = []
    tmpdirname = tempfile.mkdtemp()
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        outp = os.path.join(tmpdirname, f"{len(img_paths):06d}.jpg")
        cv2.imwrite(outp, frame)
        img_paths.append(outp)
    cap.release()
    return img_paths, tmpdirname


def downsample_label_map_majority_vote(label_map, num_classes, target_size):
    """
    label_map: torch.LongTensor (H,W) -> returns np.int32 (H_out,W_out)
    """
    H, W = label_map.shape
    H_out, W_out = target_size
    if (H, W) == (H_out, W_out):
        return label_map.cpu().numpy().astype(np.int32)

    x = label_map.unsqueeze(0).long()
    one_hot = F.one_hot(x, num_classes=num_classes).permute(0, 3, 1, 2).float()
    scale_h = max(1, H // H_out)
    scale_w = max(1, W // W_out)
    pooled = F.avg_pool2d(one_hot, kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w))
    upsampled = F.interpolate(pooled, size=target_size, mode="nearest")
    out = torch.argmax(upsampled, dim=1)
    return out.squeeze(0).cpu().numpy().astype(np.int32)


def run_inference(args):
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    add_path_to_dust3r(args.model_path)
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference
    from viser_utils import PointCloudViewer, generate_pastel_color_map, map_mask_to_3d_points
    from src.dust3r.utils.image import load_images
    from src.dust3r.utils.camera import pose_encoding_to_camera

    # Prepare input frames.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, target_fps=args.fps)
    if not img_paths:
        print(f"No images found in {args.seq_path}")
        return

    # prompt
    with open(os.path.join(args.prompt_path, "prompt.json"), "r", encoding="utf-8") as f:
        prompt = str(json.load(f).get("prompt", ""))

    # Infer the video id from the mask directory layout (.../<VID>/<pid>/FL2_mask_data).
    video_id = os.path.basename(os.path.dirname(os.path.dirname(args.mask_path)))
    # Resolve the prompt id.
    prompt_id = args.prompt_id.strip() or os.path.basename(os.path.dirname(args.mask_path))

    # Create prompt-specific output directories.
    prompt_out_root = os.path.join(args.output_dir, str(prompt_id))
    csv_dir     = os.path.join(prompt_out_root, "pts3ds_seg_csv")
    json_dir    = os.path.join(prompt_out_root, "pts3ds_seg_dict")
    color_dir   = os.path.join(prompt_out_root, "color")
    cam_dir     = os.path.join(prompt_out_root, "camera_center")
    speeds_dir  = os.path.join(prompt_out_root, "speeds")
    points_root = os.path.join(prompt_out_root, "points3d")
    for d in [csv_dir, json_dir, color_dir, cam_dir, speeds_dir, points_root]:
        _ensure_dir(d)

    # Load the image sequence.
    print(f"Found {len(img_paths)} images in {args.seq_path}. Preparing input views...")
    images = load_images(img_paths, size=args.size)
    views = []
    for i in range(len(images)):
        views.append({
            "img": images[i]["img"],
            "ray_map": torch.full(
                (images[i]["img"].shape[0], 6, images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load the model and run inference.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    print("Running inference...")
    t0 = time.time()
    outputs, state_args = inference(views, model, device)
    t1 = time.time()
    print(f"Inference completed in {t1 - t0:.2f} s (avg {(t1 - t0) / len(views):.2f} s/frame).")

    # Collect model outputs.
    pred_list = outputs["pred"]
    B = len(pred_list)
    pts3ds_world = [p["pts3d_in_other_view"].cpu().numpy()[0] for p in pred_list]  # (H,W,3)
    colors = [0.5 * (v["img"].permute(0, 2, 3, 1).cpu().numpy()[0] + 1.0) for v in outputs["views"]]  # (H,W,3)
    conf = [p["conf"].cpu().numpy()[0] if "conf" in p else None for p in pred_list]  # (H,W) or None

    pr_poses = [pose_encoding_to_camera(p["camera_pose"].clone()).cpu().numpy()[0] for p in pred_list]  # (4,4)
    H, W, _ = pts3ds_world[0].shape
    focal = np.full((B,), H / 2.0, dtype=np.float32)
    pp = np.stack([np.full((B,), W // 2), np.full((B,), H // 2)], axis=1).astype(np.float32)
    cam_dict = {
        "focal": focal,
        "pp": pp,
        "R": np.stack([P[:3, :3] for P in pr_poses], axis=0),
        "t": np.stack([P[:3, 3] for P in pr_poses], axis=0),
    }

    # Load segmentation masks and match them to the inference resolution.
    mask_files = sorted(
        [f for f in os.listdir(args.mask_path) if f.endswith(".npz")],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    if len(mask_files) != B:
        print(f"[Warn] mask frames ({len(mask_files)}) != model frames ({B}). Will map min length.")
    M = min(len(mask_files), B)

    segmasks = []
    for i in range(M):
        data = np.load(os.path.join(args.mask_path, mask_files[i]))
        mask = data["mask"]  # (H0, W0)
        mask_t = torch.from_numpy(mask).long()
        mask_resized = downsample_label_map_majority_vote(
            mask_t, num_classes=int(mask.max()) + 1, target_size=(H, W)
        )
        segmasks.append(mask_resized)
    for i in range(M, B):
        segmasks.append(np.zeros((H, W), dtype=np.int32))

    # Collect summaries, tracks, and instantaneous speeds.
    rows = []
    label_tracks = {}            # lbl -> [(fid, centroid_or_None), ...]
    seen_labels = set()          # Labels that have appeared at least once.

    # Prepare the CSV schema.
    inst_csv_path = os.path.join(speeds_dir, "instant_speeds.csv")
    csv_cols = [
        "video_id", "prompt_id", "text_prompt", "label_id",
        "from_frame", "to_frame", "dt_s",
        "dist_m", "speed_mps", "speed_kmh", "speed_mph",
        "present_from", "present_to",
        "cx_from", "cy_from", "cz_from", "cx_to", "cy_to", "cz_to"
    ]
    csv_rows = []

    # Iterate over frames.
    for f_id in range(B):
        # # Optional: save per-frame color images.
        # if args.save_summaries:
        #     iio.imwrite(os.path.join(color_dir, f"{f_id:06d}.png"), (colors[f_id] * 255).astype(np.uint8))

        # Optionally save the camera center for the current frame.
        cam_pos = (pr_poses[f_id] @ np.array([0, 0, 0, 1], dtype=np.float32))[:3]
        if args.save_summaries:
            np.save(os.path.join(cam_dir, f"{f_id:06d}_cam_center.npy"), cam_pos.astype(np.float32))

        # Map 3D points to instance labels.
        pts, labels = map_mask_to_3d_points(
            pts3d_hw3=pts3ds_world[f_id],
            segmask_hw=segmasks[f_id],
            conf_hw=conf[f_id] if conf[f_id] is not None else None,
            conf_thr=args.vis_threshold if conf[f_id] is not None else None,
            drop_background=True,
        )
        uniq = np.unique(labels) if pts.size > 0 else np.array([], dtype=np.int32)
        present_labels = set(int(x) for x in uniq.tolist())

        # Track every label that has appeared so far.
        seen_labels |= present_labels
        # Initialize per-label tracks when first seen.
        for lbl in seen_labels:
            if lbl not in label_tracks:
                label_tracks[lbl] = []

        # Save instance points for every label that has appeared at least once.
        if args.save_instance_points and (f_id % max(1, args.points_stride) == 0):
            for lbl in sorted(seen_labels):
                inst_dir = os.path.join(points_root, str(int(lbl)))
                _ensure_dir(inst_dir)
                outp = os.path.join(inst_dir, f"{f_id:06d}.npy")
                if lbl in present_labels and pts.size > 0:
                    seg_pts = pts[labels == lbl]
                    arr = seg_pts.astype(np.float32) if seg_pts.size > 0 else np.empty((0, 3), np.float32)
                else:
                    arr = np.empty((0, 3), np.float32)
                np.save(outp, arr)

        # Compute centroids and frame-level summaries.
        centroids = {}
        for lbl in present_labels:
            seg_pts = pts[labels == lbl]
            if seg_pts.size > 0:
                c = seg_pts.mean(axis=0)
                centroids[lbl] = c
        # Append to each seen label track, using None when absent.
        for lbl in sorted(seen_labels):
            c = centroids.get(lbl, None)
            label_tracks[lbl].append((f_id, None if c is None else c.tolist()))

        # Optionally save per-frame summary files.
        if args.save_summaries:
            row = {
                "frame_id": f_id,
                "video_id": video_id,
                "prompt_id": str(prompt_id),
                "text_prompt": prompt,
                "labels_present": sorted(list(present_labels)),
            }
            rows.append(row)
            with open(os.path.join(json_dir, f"{f_id:06d}.json"), "w") as fj:
                json.dump(row, fj, indent=2)

    # Backfill pre-appearance frames with empty arrays in the points3d dump.
    if args.save_instance_points:
        for lbl, seq in label_tracks.items():
            first_present = next((fid for fid, c in seq if c is not None), None)
            if first_present is None:
                # Skip labels that never produced real points to avoid excessive empty files.
                continue
            inst_dir = os.path.join(points_root, str(int(lbl)))
            _ensure_dir(inst_dir)
            for fid in range(0, first_present):
                outp = os.path.join(inst_dir, f"{fid:06d}.npy")
                if not os.path.exists(outp):
                    np.save(outp, np.empty((0, 3), np.float32))

    # ─────────────────────────────────────────────────────────
    # Pad all label tracks to length B before CSV generation.
    full_tracks = {}
    for lbl, seq in label_tracks.items():
        by_fid = {fid: c for fid, c in seq}
        full_seq = []
        for fid in range(B):
            full_seq.append((fid, by_fid.get(fid, None)))
        full_tracks[lbl] = full_seq

    # Write instantaneous speeds for every frame pair. Missing values stay as "None".
    for lbl, seq in full_tracks.items():
        prev_fid, prev_c = seq[0]
        for idx in range(1, len(seq)):
            fid, c = seq[idx]
            dt = (fid - prev_fid) / float(args.fps)
            present_from = 1 if prev_c is not None else 0
            present_to   = 1 if c is not None else 0

            if present_from and present_to:
                c0 = np.asarray(prev_c, dtype=np.float32)
                c1 = np.asarray(c, dtype=np.float32)
                dist = float(np.linalg.norm(c1 - c0))
                speed = dist / dt if dt > 0 else 0.0
                row = {
                    "video_id": video_id,
                    "prompt_id": str(prompt_id),
                    "text_prompt": prompt,
                    "label_id": int(lbl),
                    "from_frame": int(prev_fid),
                    "to_frame": int(fid),
                    "dt_s": float(dt),
                    "dist_m": dist,
                    "speed_mps": speed,
                    "speed_kmh": speed * 3.6,
                    "speed_mph": speed * 2.236936,
                    "present_from": 1,
                    "present_to": 1,
                    "cx_from": float(c0[0]), "cy_from": float(c0[1]), "cz_from": float(c0[2]),
                    "cx_to":   float(c1[0]), "cy_to":   float(c1[1]), "cz_to":   float(c1[2]),
                }
            else:
                row = {
                    "video_id": video_id,
                    "prompt_id": str(prompt_id),
                    "text_prompt": prompt,
                    "label_id": int(lbl),
                    "from_frame": int(prev_fid),
                    "to_frame": int(fid),
                    "dt_s": float(dt),
                    "dist_m": "None",
                    "speed_mps": "None",
                    "speed_kmh": "None",
                    "speed_mph": "None",
                    "present_from": present_from,
                    "present_to": present_to,
                    "cx_from": "None" if prev_c is None else float(prev_c[0]),
                    "cy_from": "None" if prev_c is None else float(prev_c[1]),
                    "cz_from": "None" if prev_c is None else float(prev_c[2]),
                    "cx_to":   "None" if c is None else float(c[0]),
                    "cy_to":   "None" if c is None else float(c[1]),
                    "cz_to":   "None" if c is None else float(c[2]),
                }
            csv_rows.append(row)
            prev_fid, prev_c = fid, c

    # Save the CSV output.
    if csv_rows:
        df = pd.DataFrame(csv_rows, columns=csv_cols)
        df.to_csv(inst_csv_path, index=False)
        print(f"[Info] Saved instant speeds to {inst_csv_path}")
    else:
        print("[Warn] No rows for instant speeds CSV (no tracks?)")

    # # Optional viewer launch can be added here when not running headless.
    # try:
    #     color_map = generate_pastel_color_map(max_labels=50)
    #     viewer = PointCloudViewer(
    #         model=model,
    #         state_args=state_args,
    #         pc_list=pts3ds_world,
    #         color_list=colors,
    #         conf_list=conf,
    #         seg_mask_list=segmasks,
    #         cam_dict=cam_dict,
    #         label_color_map=color_map,
    #         device=device,
    #         show_camera=True,
    #         vis_threshold=args.vis_threshold,
    #         size=args.size,
    #     )
    #     viewer.run()
    # except Exception as e:
    #     print(f"[Viewer] skipped: {e}")


def main():
    args = parse_args()
    _ensure_dir(args.output_dir)
    run_inference(args)


if __name__ == "__main__":
    main()
