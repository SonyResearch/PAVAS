#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL — VGGSound val → Physics-aware (MOVING) object→state/action JSON map
Slurm-friendly, single-worker, robust:
- __ready__ after model load (init time excluded from timeout)
- __start__ when a job actually begins (timeout starts here)
- job_id matching to drop stale/late results
- skip-on-timeout + worker respawn + queue drain
- split-aware output layout (from --list_file name)
- resume support

Outputs:
  <out_root_base>/<auto_subdir>/<VID>/physics_moving.json
Skip log:
  <out_root_base>/<auto_subdir>/_skipped.txt
"""

import os, sys, re, json, time, argparse, csv
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

import random, numpy as np, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ────────── seeds / determinism ──────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ────────── dataset paths ──────────
VIDEO_DIR = Path(os.environ.get("PAVAS_VGGSOUND_VIDEO_DIR", "/path/to/vggsound/video"))

# ────────── globals (set in main) ──────────
OUT_ROOT: Path = Path(".")
SKIPPED_LOG: Path = OUT_ROOT / "_skipped.txt"

# ────────── generation / decoding ──────────
MIN_PIXELS      = 4 * 28 * 28
MAX_PIXELS      = 512 * 28 * 28
MAX_NEW_TOKENS  = 192
MIN_NEW_TOKENS  = 16

# ────────── prompt ──────────
def build_prompt_moving_objects() -> str:
    return (
        "Return ONLY a plain JSON object (no markdown, no code fences, no extra text) "
        "that maps the MOVING OBJECTS in the video to their visible STATE or ACTION.\n\n"
        "DEFINITION OF MOVING OBJECT (sequence-level):\n"
        "- Determine motion over the entire clip (not frame-by-frame).\n"
        "- Count as moving if ANY of the following holds across the clip:\n"
        "  (a) Translation: the object's position changes relative to the scene/background (parallax-aware).\n"
        "  (b) Rotation/orientation change: sustained rotation or turning (e.g., door opening, car turning).\n"
        "  (c) Articulation or deformation driven by the object or its operator "
        "(e.g., walking, running, jumping, swinging, throwing, opening, closing, lifting, pushing, pulling, "
        "rolling, bouncing, striking, strumming, typing, pedaling).\n"
        "- Do NOT count as moving: apparent motion caused only by camera pan/zoom/tilt/shake when the object is fixed; "
        "microscopic flicker/aliasing; motion that is ambiguous or below perceptual threshold.\n\n"
        "FORMAT:\n"
        "- keys = lowercase short noun phrases for moving objects; you should include simple attributes to distinguish instances "
        "(e.g., runner in striped shirt, red car).\n"
        "- values = short gerund verb phrase (1-3 words), optionally with a direct object (e.g., drifting; pushing cart; opening door).\n"
        "EXCLUDE as keys (even if visibly moving):\n"
        "- Continuous media, such as water bodies (ocean, river, waves), atmospheric phenomena (smoke, fog, clouds), "
        "fire (flames, wildfire), precipitation (rain, snow, hail), wind/storms (gusts, tornado, hurricane, blizzard), "
        "or volcanic matter (lava, ash).\n"
        "- Clothing/accessories as standalone keys (shirts, hats, logos, patterns). "
        "Use them only as attributes to disambiguate an entity (e.g., runner in striped shirt).\n\n"
        'OUTPUT: {"object":"state/action", ...}\n'
        "Examples:\n"
        '{"red car":"drifting","black pickup truck":"colliding"}\n'
        '{"basketball":"bouncing","player":"running"}\n'
        '{"door":"opening","person":"pushing"}\n'
        '{"runner in striped shirt":"sprinting","black dog":"chasing"}\n'
        '{"shopping cart":"rolling","man":"pushing"}'
    )

# ────────── helpers ──────────
SPLIT_RE = re.compile(r"split_(\d+)", re.IGNORECASE)

def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def _extract_balanced_json_objects(text: str) -> list[str]:
    objs, stack, start = [], 0, -1
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    objs.append(text[start:i+1]); start = -1
    return objs

def coerce_json_object(text: str):
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    for blob in _extract_balanced_json_objects(text):
        if "```" in blob or blob.count(":") == 0:
            continue
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}

_BAD_TOKEN_RE = re.compile(r"[`{}[\]<>\n]|```", re.I)
_ALLOWED_CHARS_RE = re.compile(r"^[a-z0-9 ,.'\-/]+$", re.I)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _suspicious_key(s: str) -> bool:
    if not s or len(s) > 80: return True
    if _BAD_TOKEN_RE.search(s): return True
    if not _ALLOWED_CHARS_RE.match(s): return True
    if not re.search(r"[a-z]", s): return True
    return False

def _suspicious_val(s: str) -> bool:
    if not s or len(s) > 40: return True
    if _BAD_TOKEN_RE.search(s): return True
    if not _ALLOWED_CHARS_RE.match(s): return True
    return False

def sanitize_keys_values(obj_map: dict) -> dict:
    out = {}
    for k, v in (obj_map or {}).items():
        nk, nv = _norm(k), _norm(v)
        if _suspicious_key(nk) or _suspicious_val(nv):
            continue
        if nv in {"stationary", "static", "idle", "standing still"}:
            continue
        out[nk] = nv
    return out

UNBOUNDED_MEDIA = {
    "ocean","sea","river","lake","pond","stream","creek","brook",
    "waterfall","waves","wave","surf","tide","current",
    "smoke","fog","steam","mist","haze","smog","vapor","vapour","cloud","clouds","plume",
    "fire","flame","flames","blaze","wildfire","inferno","embers","sparks",
    "rain","drizzle","snow","snowfall","sleet","hail","graupel",
    "wind","gust","gusts","tornado","hurricane","typhoon","cyclone","storm","storms","blizzard",
    "sandstorm","dust storm","duststorm","squall",
    "lava","ash","ash cloud","ashcloud",
}

def normalize_only(obj_map: dict) -> dict:
    return {normalize_text(k): normalize_text(v) for k, v in (obj_map or {}).items() if k and v}

def filter_unbounded_only(obj_map: dict) -> dict:
    return {k: v for k, v in (obj_map or {}).items() if k not in UNBOUNDED_MEDIA}

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log_skip(msg: str):
    line = f"{datetime.now().isoformat(timespec='seconds')} | {msg}\n"
    with SKIPPED_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"[SKIP] {msg}")

def read_list_file(path: Path):
    items = []
    suf = path.suffix.lower()
    if suf == ".csv":
        with path.open("r", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                if row: items.append(Path(row[0]))
    elif suf == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                v = rec.get("video")
                if v: items.append(Path(v))
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                v = line.strip()
                if v: items.append(Path(v))
    return items

def infer_split_subdir(list_file: str, out_subdir_arg: str) -> str:
    if out_subdir_arg:
        return out_subdir_arg.strip()
    if list_file:
        m = SPLIT_RE.search(Path(list_file).name)
        if m:
            return f"split_{m.group(1)}_json"
    return "default_json"

def drain_queue(q: mp.Queue):
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass

# ────────── worker process ──────────
def worker_loop(task_q: mp.Queue, res_q: mp.Queue, device_map="auto"):
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
            device_map=device_map
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        res_q.put(("__ready__",))
    except Exception as e:
        res_q.put(("__init_error__", str(e)))
        return

    def build_video_messages(video_path: str, prompt_text: str):
        return [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text",  "text":  prompt_text},
            ],
        }]

    @torch.inference_mode()
    def run_once(video_path: str, prompt_text: str) -> str:
        msgs = build_video_messages(video_path, prompt_text)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        dec = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return dec[0] if dec else ""

    while True:
        item = task_q.get()
        if item is None:
            break
        job_id, vid_path, prompt = item
        res_q.put(("__start__", job_id))
        try:
            raw = run_once(vid_path, prompt)
            res_q.put(("ok", job_id, vid_path, raw))
        except Exception as e:
            res_q.put(("error", job_id, vid_path, repr(e)))

# ────────── main orchestrator ──────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default="*.mp4", help="glob under the configured video directory when no list_file is provided")
    ap.add_argument("--limit", type=int, default=0, help="first N (0=all)")
    ap.add_argument("--timeout", type=int, default=12, help="per-video timeout (after __start__)")
    ap.add_argument("--prestart_timeout", type=int, default=6, help="enqueue→__start__ timeout (0=off)")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--list_file", type=str, default="")
    ap.add_argument("--out_root_base", type=str, required=True)
    ap.add_argument("--out_subdir", type=str, default="")
    ap.add_argument("--resume", action="store_true", help="skip if json exists")
    args = ap.parse_args()

    # Build list
    if args.list_file:
        list_path = Path(args.list_file)
        vids = read_list_file(list_path)
    else:
        vids = sorted([p for p in VIDEO_DIR.glob(args.pattern) if p.is_file()])
    if args.limit and args.limit > 0:
        vids = vids[:args.limit]

    # Normalize to absolute paths (relative -> VIDEO_DIR / rel)
    vids = [p if Path(p).is_absolute() else (VIDEO_DIR / p) for p in vids]

    # Output dirs
    out_root_base = Path(args.out_root_base)
    auto_subdir = infer_split_subdir(args.list_file, args.out_subdir)
    out_root = out_root_base / auto_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    global OUT_ROOT, SKIPPED_LOG
    OUT_ROOT = out_root
    SKIPPED_LOG = out_root / "_skipped.txt"

    print(f"[INFO] videos: {len(vids)}")
    print(f"[INFO] output: {OUT_ROOT}")

    P_MOV = build_prompt_moving_objects()

    task_q: mp.Queue = mp.Queue(maxsize=1)
    res_q: mp.Queue = mp.Queue(maxsize=16)

    def start_worker():
        p = mp.Process(target=worker_loop, args=(task_q, res_q, args.device_map), daemon=True)
        p.start()
        # wait for __ready__
        while True:
            kind, *payload = res_q.get()
            if kind == "__ready__":
                break
            if kind == "__init_error__":
                err = payload[0] if payload else "unknown"
                log_skip(f"__init_error__ | {err}")
                sys.exit(1)
        return p

    worker = start_worker()

    for i, vp in enumerate(vids, 1):
        vp = Path(vp)
        vid_id = vp.stem
        out_dir = OUT_ROOT / vid_id
        out_path = out_dir / "physics_moving.json"
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.resume and out_path.exists():
            if i % 10 == 0:
                print(f"[OK] {i}/{len(vids)} (cached) — last: {vid_id}")
            continue

        job_id = f"{i}:{vid_id}:{int(time.time()*1000)}"
        try:
            task_q.put((job_id, str(vp), P_MOV))
        except Exception as e:
            log_skip(f"{vp.name} | enqueue_error: {e}")
            continue

        t_enq = time.time()
        t_start = None
        result_kind = None
        result_payload = None

        while True:
            now = time.time()
            # prestart timeout (before __start__)
            if t_start is None and args.prestart_timeout > 0 and (now - t_enq > args.prestart_timeout):
                log_skip(f"{vp.name} | prestart timeout {args.prestart_timeout}s → SKIP")
                try:
                    worker.terminate(); worker.join(timeout=5)
                except Exception:
                    pass
                drain_queue(task_q); drain_queue(res_q)
                worker = start_worker()
                result_kind = "skip"
                break

            # main timeout (after __start__)
            if t_start is not None and (now - t_start > args.timeout):
                log_skip(f"{vp.name} | timeout {args.timeout}s → SKIP")
                try:
                    worker.terminate(); worker.join(timeout=5)
                except Exception:
                    pass
                drain_queue(task_q); drain_queue(res_q)
                worker = start_worker()
                result_kind = "skip"
                break

            # pump messages
            try:
                kind, *payload = res_q.get(timeout=0.2)
            except Exception:
                continue

            if kind == "__start__":
                r_job = payload[0]
                if r_job == job_id:
                    t_start = time.time()
                continue

            if kind in ("ok", "error"):
                r_job = payload[0]
                if r_job != job_id:
                    # stale/late result → drop
                    continue
                result_kind = kind
                result_payload = payload[1:]  # (vid_path, raw) or (vid_path, err)
                break

            if kind == "__init_error__":
                err = payload[0] if payload else "unknown"
                log_skip(f"__init_error__ | {err}")
                sys.exit(1)

        # handle result
        if result_kind == "ok":
            _vid_path_from_worker, raw = result_payload
            try:
                # parsing + normalization + filtering + sanitize
                obj_map_raw = coerce_json_object(raw)
                obj_map_norm = normalize_only(obj_map_raw)
                obj_map_tmp  = filter_unbounded_only(obj_map_norm)
                obj_map      = sanitize_keys_values(obj_map_tmp)

                payload = {
                    "video": str(vp),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prompt": P_MOV,
                    "raw": raw,
                    "map": obj_map,
                }
                save_json(out_path, payload)

                if not obj_map:
                    log_skip(f"{vp.name} | sanitized to empty map (garbage output suppressed)")

            except Exception as e:
                log_skip(f"{vp.name} | save_error: {e}")

        elif result_kind == "error":
            r_vid_path, err = result_payload
            log_skip(f"{Path(r_vid_path).name} | runtime_error: {err}")

        # result_kind == "skip": already logged & worker respawned

        if i % 10 == 0:
            print(f"[OK] {i}/{len(vids)} processed — last: {vid_id}")

    # shutdown
    try:
        task_q.put(None)
    except Exception:
        pass
    try:
        worker.join(timeout=5)
    except Exception:
        pass

    print(f"[DONE] outputs under {OUT_ROOT}")
    if SKIPPED_LOG.exists():
        print(f"[INFO] skipped list: {SKIPPED_LOG}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
