#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL: VGGSound MOVING OBJECTS → mass estimation (per-video JSON)
- Input (per split):  <in_root_base>/<in_subdir>/<VID>/physics_moving.json
- Output:             <out_root_base>/<out_subdir>/<VID>/mass_estimation.json
- No tokenizer padding override; no mp4 copy.
"""

import json, re, math, time, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import random, numpy as np, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── seeds
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── vision token scale / generation
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS  = 192
MIN_NEW_TOKENS  = 16
DO_SAMPLE = True
TEMPERATURE = 0.4
TOP_P = 0.92
RETRY_ON_TRUNC = True
RETRY_MULTIPLIER = 2

OUTPUT_FILENAME = "mass_estimation.json"

def build_prompt_mass_from_moving_objects(objects_map: Dict[str, str]) -> str:
    obj_json = json.dumps(objects_map, ensure_ascii=False, separators=(",", ":"))
    examples = """
Examples:
MOVING OBJECTS: {"black pickup truck":"colliding","red car":"drifting"}
OUTPUT: {"black pickup truck":{"rationale":"Mid-size double-cab pickup; steel frame → about 1925.0 kg","weight_kg":1925.0},
         "red car":{"rationale":"Compact sedan; 4-door body → about 1410.0 kg","weight_kg":1410.0}}

MOVING OBJECTS: {"basketball":"bouncing","player":"running"}
OUTPUT: {"basketball":{"rationale":"Official size 7 ball; inflated rubber bladder → about 0.62 kg","weight_kg":0.62},
         "player":{"rationale":"Adult male; medium build → about 72.0 kg","weight_kg":72.0}}

MOVING OBJECTS: {"dog":"jumping","suitcase":"rolling"}
OUTPUT: {"dog":{"rationale":"Medium-size Labrador; lean build → about 27.0 kg","weight_kg":27.0},
         "suitcase":{"rationale":"Hard-shell carry-on, empty → about 3.8 kg","weight_kg":3.8}}

MOVING OBJECTS: {"guitar":"being strummed","chair":"falling"}
OUTPUT: {"guitar":{"rationale":"Full-size acoustic guitar with wooden body → about 2.2 kg","weight_kg":2.2},
         "chair":{"rationale":"Plastic molded chair, lightweight → about 3.5 kg","weight_kg":3.5}}

MOVING OBJECTS: {"quadcopter drone":"hovering","bicycle":"coasting"}
OUTPUT: {"quadcopter drone":{"rationale":"Prosumer quadcopter with gimbal; typical takeoff mass → about 0.90 kg","weight_kg":0.90},
         "bicycle":{"rationale":"Aluminum road bike, no racks/fenders → about 9.5 kg","weight_kg":9.5}}
""".strip()

    return f"""
You will be shown a short VIDEO.
From this video, MOVING OBJECTS and their visible STATE/ACTION are provided as a JSON map.

MOVING OBJECTS (object → state/action):
{obj_json}

TASK
For every key in MOVING OBJECTS, estimate its mass in kilograms and return exactly one JSON object of the form:
{{"object":{{"rationale":"<concise explanation ending with the same number>","weight_kg":<number>}}, ...}}

POLICY
- Use only the keys given in MOVING OBJECTS; do not invent, rename, split, or merge keys.
- Include every key exactly once, in the same order as in MOVING OBJECTS.
- Keys must match MOVING OBJECTS verbatim, character-for-character.
- "weight_kg" must always be a numeric value (no units, no text).
- Rationales should be concise, reference visible attributes (e.g., material, dimensions, body build), and end with the same number reported in "weight_kg".
- Avoid round anchor numbers unless clearly justified.
- Base estimates on visible material, thickness, density, and overall volume in the video. For standardized items use specs (convert lb→kg by ÷2.20462).
- Return only the final JSON object. No markdown, no code fences, no extra text.

{examples}
""".strip()

def build_messages_video(video_path: str, text_prompt: str):
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(video_path), "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text",  "text": text_prompt},
        ],
    }]

@torch.inference_mode()
def qwen_infer_video(model, processor, video_path: str, prompt_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = build_messages_video(video_path, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"; inputs = inputs.to(device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": DO_SAMPLE, "temperature": TEMPERATURE, "top_p": TOP_P}
    try: gen_kwargs["min_new_tokens"] = MIN_NEW_TOKENS
    except Exception: pass
    out_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    dec = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return dec[0] if dec else ""

JSON_BLOCK_RE = re.compile(r'\{.*\}', re.DOTALL)

def extract_json_block(text: str) -> str:
    if not text: return ""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start: return text[start:end+1].strip()
    m = JSON_BLOCK_RE.search(text or ""); return m.group(0).strip() if m else ""

def _coerce_weight(val: Any) -> Tuple[Optional[float], bool]:
    if val is None: return None, True
    if isinstance(val, (int, float)):
        try:
            v = float(val); 
            return (v, True) if math.isfinite(v) else (None, False)
        except Exception: return None, False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"null", "none", ""}: return None, True
        try:
            v = float(s); 
            return (v, True) if math.isfinite(v) else (None, False)
        except Exception: return None, False
    return None, False

def normalize_output(raw_text: str, objects_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    block = extract_json_block(raw_text)
    try:
        obj = json.loads(block); 
        if not isinstance(obj, dict): obj = {}
    except Exception:
        obj = {}
    keys_in = list(objects_map.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys_in:
        v = obj.get(k, None)
        if isinstance(v, dict):
            rationale = v.get("rationale", "")
            w_raw = v.get("weight_kg", None)
            w, ok = _coerce_weight(w_raw)
            if not ok: rationale = (rationale or "")[:150]; w = None
            out[k] = {"rationale": rationale, "weight_kg": w}; continue
        if isinstance(v, (int, float)) or v is None or (isinstance(v, str) and v.strip().lower() in {"null", "none", ""}):
            w, ok = _coerce_weight(v)
            out[k] = {"rationale": "model returned scalar weight" if ok and w is not None else "model returned null/no weight", "weight_kg": w}
            continue
        if isinstance(v, str):
            out[k] = {"rationale": v[:160], "weight_kg": None}
        else:
            out[k] = {"rationale": "invalid output type", "weight_kg": None}
    return out

def likely_truncated(text: str, max_new_tokens: int) -> bool:
    if not text: return True
    opens, closes = text.count("{"), text.count("}")
    if opens > closes: return True
    if not text.rstrip().endswith("}"): return True
    return len(text) > 0.9 * max_new_tokens

def is_effectively_empty_map(m: Dict[str, str]) -> bool:
    if not isinstance(m, dict) or not m: return True
    for k, v in m.items():
        if str(k).strip() and str(v).strip(): return False
    return True

def find_moving_jsons(base: Path) -> List[Path]:
    # depth-1: <base>/<vid>/physics_moving.json
    return sorted([p for p in base.glob("*/physics_moving.json") if p.is_file()])

def load_moving(path: Path) -> Tuple[str, Dict[str, str]]:
    data = json.loads(path.read_text())
    video = data.get("video", "")
    objects_map = data.get("map", {})
    if not isinstance(objects_map, dict): objects_map = {}
    return video, objects_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root_base",  type=str, required=True,  help="Root dir containing split subdirs (e.g., _bow_moving_split)")
    ap.add_argument("--in_subdir",     type=str, required=True,  help="Split subdir (e.g., split_07_json)")
    ap.add_argument("--out_root_base", type=str, required=True,  help="Output root dir (e.g., _mass_from_moving_split)")
    ap.add_argument("--out_subdir",    type=str, default="",     help="Optional subdir (default: same as in_subdir)")
    ap.add_argument("--overwrite",     action="store_true",      help="Recompute even if output exists")
    args = ap.parse_args()

    in_root_base  = Path(args.in_root_base)
    in_split_dir  = in_root_base / args.in_subdir
    out_root_base = Path(args.out_root_base)
    out_subdir    = args.out_subdir.strip() or args.in_subdir
    out_split_dir = out_root_base / out_subdir
    out_split_dir.mkdir(parents=True, exist_ok=True)

    skip_log = out_root_base / "_skipped.txt"
    def log_skip(line: str):
        with skip_log.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} | {line}\n")

    print(f"[INFO] input split dir : {in_split_dir}")
    print(f"[INFO] output split dir: {out_split_dir}")

    # model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    items = find_moving_jsons(in_split_dir)
    print(f"[INFO] found {len(items)} physics_moving.json files")

    n_ok, n_skip, n_retry = 0, 0, 0
    t0 = time.time()

    for moving_path in items:
        try:
            video_path, objects_map = load_moving(moving_path)
            if not video_path or not Path(video_path).exists():
                msg = f"video missing for {moving_path}"
                print(f"[WARN] {msg}"); log_skip(msg); n_skip += 1; continue

            if is_effectively_empty_map(objects_map):
                msg = f"empty-or-degenerate MOVING map for {moving_path}"
                print(f"[WARN] {msg}"); log_skip(msg); n_skip += 1; continue

            vid_id  = Path(video_path).stem
            out_dir = out_split_dir / vid_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / OUTPUT_FILENAME

            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] exists → {out_path}")
                log_skip(f"exists → {out_path}")
                n_skip += 1; continue

            prompt  = build_prompt_mass_from_moving_objects(objects_map)
            out_txt = qwen_infer_video(model, processor, video_path, prompt, MAX_NEW_TOKENS)

            if RETRY_ON_TRUNC and likely_truncated(out_txt, MAX_NEW_TOKENS):
                n_retry += 1
                out_txt2 = qwen_infer_video(model, processor, video_path, prompt, MAX_NEW_TOKENS * RETRY_MULTIPLIER)
                if out_txt2 and len(out_txt2) >= len(out_txt): out_txt = out_txt2

            normalized = normalize_output(out_txt, objects_map)
            payload = {"video": video_path, "prompt": prompt, "map": objects_map, "mass_estimation": normalized}
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            n_ok += 1
            print(f"[OK] {moving_path.parent.name} → {out_path}")

        except Exception as e:
            msg = f"{moving_path}: {e}"
            print(f"[ERROR] {msg}"); log_skip(f"exception {msg}"); n_skip += 1

    print(f"[STATS] ok={n_ok}, skipped={n_skip}, retries={n_retry}, elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
