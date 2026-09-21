#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Set the repository root, PYTHONPATH, and CWD before any local imports.
import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
# =======================================================================

import json
import time
import copy
import argparse
import logging
from multiprocessing import Process, set_start_method

import cv2
import numpy as np
import torch
from PIL import Image

# Force math/eager attention kernels for compatibility with this Transformers stack.
try:
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
except Exception:
    pass

# SAM2 / Florence2 dependencies resolved from the vendored repository layout.
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM

# --- your utils ---
from utils.common_utils import CommonUtils
from utils.video_utils import create_video_from_images
from utils.track_utils import sample_points_from_masks
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

# ---------------- CONFIG ----------------
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"

# Hydra config name relative to the vendored package root.
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l"

# SAM2 checkpoint file
SAM2_CHECKPOINT = os.environ.get(
    "PAVAS_GSAM2_CKPT",
    str(REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"),
)
if not os.path.isfile(SAM2_CHECKPOINT):
    raise FileNotFoundError(
        f"SAM2_CHECKPOINT not found: {SAM2_CHECKPOINT}. "
        "Set PAVAS_GSAM2_CKPT or place the checkpoint under grounded_sam2/checkpoints/."
    )

TASK_PROMPT = "<OPEN_VOCABULARY_DETECTION>"
# ----------------------------------------


def run_inference_on_video(
    folder_path, class_strings, file_name, output_dir,
    predictor, video_predictor, florence2_model, florence2_processor,
    device, fps, save_box_viz: bool, save_video: bool
):
    frame_names = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    for idx, class_name in enumerate(class_strings):
        CommonUtils.creat_dirs(output_dir)
        class_output_dir = os.path.join(output_dir, os.path.basename(folder_path), str(idx))
        CommonUtils.creat_dirs(class_output_dir)

        mask_data_dir = os.path.join(class_output_dir, "FL2_mask_data")
        json_data_dir = os.path.join(class_output_dir, "FL2_json_data")
        text_prompt_dir = os.path.join(class_output_dir, "FL2_text_prompt")
        result_dir2 = os.path.join(class_output_dir, "FL2_Box_Result")

        # Create only the directories required for this class run.
        for d in [mask_data_dir, json_data_dir, text_prompt_dir] + ([result_dir2] if save_box_viz else []):
            CommonUtils.creat_dirs(d)

        # SAM2 video state
        inference_state = video_predictor.init_state(
            video_path=folder_path, offload_video_to_cpu=True, async_loading_frames=True
        )

        step = 20
        sam2_masks = MaskDictionaryModel()
        objects_count = 0

        for start_frame_idx in range(0, len(frame_names), step):
            logging.info(f"[{file_name}] start_frame_idx = {start_frame_idx}")

            img_path = os.path.join(folder_path, frame_names[start_frame_idx])
            image = Image.open(img_path).convert("RGB")
            prompt = TASK_PROMPT + class_name

            # Florence2 detection
            inputs = florence2_processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
            with torch.inference_mode():
                generated_ids = florence2_model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    pixel_values=inputs["pixel_values"].to(device),
                    max_new_tokens=1024, num_beams=3
                )
            generated_text = florence2_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = florence2_processor.post_process_generation(
                generated_text, task=TASK_PROMPT, image_size=image.size
            )

            predictor.set_image(np.array(image))
            results = parsed_answer.get(TASK_PROMPT, {})
            input_boxes = np.array(results.get("bboxes", []))
            class_names = results.get("bboxes_labels", [])

            # no boxes → save empty mask frame for traceability
            if input_boxes.shape[0] == 0 or input_boxes.ndim != 2 or input_boxes.shape[1] != 4:
                logging.warning(f"Florence2: no/invalid boxes (file: {file_name}, class: {class_name})")
                image_base_name = frame_names[start_frame_idx].split(".")[0]
                mask_name = f"mask_{image_base_name}.npy"

                dummy_image = np.array(image)
                H, W = dummy_image.shape[:2]
                empty_mask = np.zeros((H, W), dtype=np.uint8)

                np.save(os.path.join(mask_data_dir, mask_name), empty_mask)

                with open(os.path.join(json_data_dir, mask_name.replace(".npy", ".json")), "w") as f:
                    json.dump({
                        "labels": {},
                        "mask_name": mask_name,
                        "mask_height": H,
                        "mask_width": W,
                        "promote_type": "mask",
                    }, f)

                with open(os.path.join(text_prompt_dir, "prompt.json"), "w") as f:
                    json.dump({"prompt": class_name}, f)
                continue

            # SAM2 image predictor (boxes → masks)
            with torch.inference_mode():
                masks, scores, logits = predictor.predict(box=input_boxes, multimask_output=False)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            image_base_name = frame_names[start_frame_idx].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type="mask", mask_name=f"mask_{image_base_name}.npy")
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks, device=device),
                box_list=torch.tensor(input_boxes, device=device),
                label_list=class_names,
            )

            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count
            )

            # SAM2 video propagation
            video_predictor.reset_state(inference_state)
            for object_id, object_info in mask_dict.labels.items():
                _frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state, start_frame_idx, object_id, object_info.mask
                )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx
            ):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(
                        instance_id=out_obj_id,
                        mask=out_mask[0],
                        class_name=mask_dict.get_target_class_name(out_obj_id),
                    )
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info

                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            # write per-frame outputs
            for frame_idx, frame_masks_info in video_segments.items():
                mask_img = np.zeros((frame_masks_info.mask_height, frame_masks_info.mask_width), dtype=np.uint8)
                for obj_id, obj_info in frame_masks_info.labels.items():
                    mask_np = obj_info.mask.detach().float().cpu().numpy() > 0.5
                    if mask_np.shape != (frame_masks_info.mask_height, frame_masks_info.mask_width):
                        resized_mask = cv2.resize(
                            mask_np.astype(np.uint8),
                            (frame_masks_info.mask_width, frame_masks_info.mask_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        resized_mask = mask_np
                    mask_img[resized_mask] = obj_id

                # save as .npy (will compress to .npz later)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

                with open(os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json")), "w") as f:
                    json.dump(frame_masks_info.to_dict(), f)

                with open(os.path.join(text_prompt_dir, "prompt.json"), "w") as f:
                    json.dump({"prompt": class_name}, f)

        # draw & video & compress
        if not os.path.exists(mask_data_dir) or not any(Path(mask_data_dir).glob("*.npy")):
            logging.warning(f"mask directory empty: {mask_data_dir}")
            continue
        else:
            if save_box_viz:
                CommonUtils.draw_masks_and_box_with_supervision(folder_path, mask_data_dir, json_data_dir, result_dir2)
            if save_video and save_box_viz:
                create_video_from_images(result_dir2, os.path.join(class_output_dir, "FL2.mp4"), frame_rate=fps)

            # .npy → .npz
            for npy_file in Path(mask_data_dir).glob("*.npy"):
                data = np.load(npy_file)
                np.savez_compressed(npy_file.with_suffix(".npz"), mask=data)
                os.remove(npy_file)


def worker_subrange(args, keys, sub_start, sub_end):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    # models
    video_predictor = build_sam2_video_predictor(SAM2_CONFIG_NAME, SAM2_CHECKPOINT)
    sam2_image_model = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    florence2_model = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_MODEL_ID,
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="eager",
    ).eval().to(device)
    florence2_processor = AutoProcessor.from_pretrained(
        FLORENCE2_MODEL_ID, local_files_only=False, trust_remote_code=True
    )

    # load manifest (already opened in main, but cheap enough)
    with open(args.text, "r") as f:
        json_data = json.load(f)

    # per-worker log file
    log_path = os.path.join(
        args.output, f"FL2_train_index_log_{sub_start}_{sub_end}{('_' + args.run_tag) if args.run_tag else ''}.txt"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # basic logger (file+stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    for idx in range(sub_start, min(sub_end, len(keys))):
        key = keys[idx]
        folder_name = (Path(key).parts[-1])[:-4] if key.endswith(".mp4") else Path(key).parts[-1]
        folder_path = os.path.join(args.image_root, folder_name)

        if not os.path.exists(folder_path):
            logging.warning(f"image folder missing: {folder_path}")
            continue

        prompt_str = json_data[key].get("final output", "")
        if prompt_str.strip() == "":
            logging.error(f"empty JSON string: ({key})")
            continue

        try:
            # normalize prompt dict
            if isinstance(prompt_str, str):
                try:
                    prompt_str = json.loads(prompt_str)
                except Exception:
                    pass
            parsed = prompt_str if isinstance(prompt_str, dict) else json.loads(prompt_str)
            class_strings = list(parsed.keys())

            run_inference_on_video(
                folder_path, class_strings, key, args.output,
                image_predictor, video_predictor,
                florence2_model, florence2_processor,
                device, args.fps,
                save_box_viz=not args.no_box_viz,
                save_video=not args.no_video,
            )
        except Exception as e:
            logging.exception(f"JSON parse/inference error ({key}): {e}")
            continue

        with open(log_path, "a") as log_f:
            log_f.write(f"{idx}\t{key}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="manifest json (this split)")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)

    # NEW: control worker fanout for Slurm (1 recommended)
    parser.add_argument("--num_workers", type=int, default=1)

    # NEW: tag for log/err naming (split_XX)
    parser.add_argument("--run_tag", type=str, default="")

    # Optional flags to disable visualization artifacts.
    parser.add_argument("--no_box_viz", action="store_true", help="Do NOT save FL2_Box_Result images")
    parser.add_argument("--no_video", action="store_true", help="Do NOT save FL2.mp4")

    args = parser.parse_args()

    set_start_method("spawn", force=True)

    # Error log per run. Set KEEP_STDERR=1 if you also want stderr on the console.
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    os.makedirs(args.output, exist_ok=True)
    error_path = os.path.join(args.output, f"FL2_error{tag_suffix}.txt")
    if os.environ.get("KEEP_STDERR", "0") != "1":
        sys.stderr = open(error_path, "a")

    with open(args.text, "r") as f:
        json_data = json.load(f)
    keys = sorted([k.strip() for k in json_data.keys()])

    total = args.end_idx - args.start_idx
    if args.num_workers <= 1:
        worker_subrange(args, keys, args.start_idx, args.end_idx)
    else:
        chunk = (total + args.num_workers - 1) // args.num_workers
        processes = []
        for i in range(args.num_workers):
            sub_start = args.start_idx + i * chunk
            sub_end = min(args.start_idx + (i + 1) * chunk, args.end_idx)
            if sub_start >= sub_end:
                continue
            p = Process(target=worker_subrange, args=(args, keys, sub_start, sub_end))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    set_start_method("spawn")
    main()
