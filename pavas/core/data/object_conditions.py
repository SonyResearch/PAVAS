from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from core.utils.io_profile import io_timer  # fine-grained object build profiling
from collections import OrderedDict
import time  # ensure available if future time-based logic added
from functools import lru_cache

# Path resolvers supporting both the old and new layouts.
@lru_cache(maxsize=200_000)
def _resolve_mass_json(mass_root: str | Path, video_id: str) -> Optional[Path]:
    root = Path(mass_root)
    direct = root / video_id / "mass_estimation.json"
    if direct.exists():
        return direct
    hits = list(root.glob(f"split*_json/{video_id}/mass_estimation.json"))
    return hits[0] if hits else None

@lru_cache(maxsize=200_000)
def _resolve_seg_prompt_dir(seg_root: str | Path, video_id: str, prompt_id: int | str) -> Optional[Path]:
    root = Path(seg_root)
    pid = str(prompt_id)
    direct = root / video_id / pid
    if (direct / "FL2_mask_data").exists():
        return direct
    hits = list(root.glob(f"split_*/{video_id}/{pid}"))
    for h in hits:
        if (h / "FL2_mask_data").exists():
            return h
    return None

@dataclass
class SampleObjectInputs:
    # per-sample
    velocity_f: Optional[torch.Tensor]     # (O, Tv, 1) float32
    mass_f:     Optional[torch.Tensor]     # (O, 1)     float32
    presence:   Optional[torch.Tensor]     # (O, Tv)    bool
    obj_patch_attn: Optional[torch.Tensor] = None  # (O, Tv, Gh*Gw) float32, optional
    label_ids: Optional[torch.Tensor] = None  # (O,) int64, label_id ordering
    prompt_ids: Optional[torch.Tensor] = None  # (O,) int64, when multiple prompt dirs merged
    velocity_missing: Optional[torch.Tensor] = None  # (O, Tv) bool: frames where object is absent (velocity=None)


@dataclass
class BatchedObjectInputs:
    # batched/padded
    velocity_f: Optional[torch.Tensor]     # (B, Omax, Tvmax, 1)
    mass_f:     Optional[torch.Tensor]     # (B, Omax, 1)
    presence:   Optional[torch.Tensor]     # (B, Omax, Tvmax)
    obj_patch_attn: Optional[torch.Tensor] = None  # (B, Omax, Tvmax, GhGw)
    label_ids: Optional[torch.Tensor] = None       # (B, Omax)
    prompt_ids: Optional[torch.Tensor] = None      # (B, Omax)
    empty_mask: Optional[torch.Tensor] = None      # (B,) bool, True if sample has 0 objects after pruning
    velocity_missing: Optional[torch.Tensor] = None  # (B, Omax, Tvmax) bool


# --- Pruning log control (module-level) ---------------------------------------------------------
_OC_PRUNE_LOG_COUNTER = 0
def _maybe_log_prune(summary: str):
    """Log pruning summary for first N batches (controlled by env MMAUDIO_OC_PRUNE_LOG_N)."""
    import os, sys
    global _OC_PRUNE_LOG_COUNTER
    limit = int(os.environ.get('MMAUDIO_OC_PRUNE_LOG_N', '20'))
    if _OC_PRUNE_LOG_COUNTER < limit:
        print(summary, file=sys.stderr)
    _OC_PRUNE_LOG_COUNTER += 1


def label_map_from_velocity_csv(instant_csv_path: str | Path) -> Dict[int, str]:
    """Extract the label_id -> text_prompt mapping from a CSV file."""
    mp: Dict[int, str] = {}
    p = Path(instant_csv_path)
    if not p.exists():
        return mp
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            lid = r.get("label_id")
            tp = r.get("text_prompt")
            if lid is None or tp is None or tp == "None":
                continue
            try:
                mp[int(lid)] = str(tp)
            except Exception:
                continue
    return mp


def load_mass_json(mass_json_path: str | Path,
                   label_id_to_object: Dict[int, str]) -> torch.Tensor:
    """Return an `(O, 1)` tensor of `weight_kg` values in label-id order. Missing values become 0.0."""
    O = len(label_id_to_object)
    out = np.zeros((O, 1), dtype=np.float32)

    p = Path(mass_json_path)
    if not p.exists():
        return torch.from_numpy(out)

    try:
        data = json.loads(p.read_text())
        est = data.get("mass_estimation", {})
        lid_sorted = sorted(label_id_to_object.keys())
        for i, lid in enumerate(lid_sorted):
            name = label_id_to_object[lid]
            w = est.get(name, {}).get("weight_kg", 0.0)
            try:
                out[i, 0] = float(w)
            except Exception:
                out[i, 0] = 0.0
    except Exception:
        pass

    return torch.from_numpy(out)  # (O,1)


def parse_velocity_csv(instant_csv_path: str | Path,
                       label_ids_sorted: Sequence[int],
                       tv: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read velocity and presence from `instant_speeds.csv`.

    Returns:
      vel: `(O, Tv, 1)` float32
      present_csv: `(O, Tv)` bool
    """
    p = Path(instant_csv_path)
    O = len(label_ids_sorted)
    if not p.exists():
        Tv = tv or 1
        z = torch.zeros((O, Tv), dtype=torch.bool)
        return torch.zeros((O, Tv, 1), dtype=torch.float32), z, z

    rows_by_label: Dict[int, List[dict]] = {lid: [] for lid in label_ids_sorted}
    max_to_frame = 0
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                lid = int(r["label_id"])
                if lid in rows_by_label:
                    rows_by_label[lid].append(r)
                tf = int(r["to_frame"])
                max_to_frame = max(max_to_frame, tf)
            except Exception:
                continue

    Tv = tv if tv is not None else (max_to_frame + 1)
    vel = np.zeros((O, Tv, 1), dtype=np.float32)
    present = np.zeros((O, Tv), dtype=np.uint8)
    missing = np.zeros((O, Tv), dtype=np.uint8)
    lid_to_idx = {lid: i for i, lid in enumerate(label_ids_sorted)}

    for lid, rows in rows_by_label.items():
        oi = lid_to_idx[lid]
        for r in rows:
            try:
                tf = int(r["to_frame"])
                if not (0 <= tf < Tv):
                    continue
                pf = int(r.get("present_from", 0)) == 1
                pt = int(r.get("present_to", 0)) == 1
                sp = r.get("speed_mps", None)
                
                # Treat valid velocity values as object presence.
                if pf and pt and sp is not None and sp != "None":
                    try:
                        sp_val = float(sp)
                        if np.isfinite(sp_val):
                            vel[oi, tf, 0] = sp_val
                            present[oi, tf] = 1
                            # missing[oi, tf] already defaults to 0.
                        else:
                            # Parsed value is inf/nan, so mark the object as absent.
                            missing[oi, tf] = 1
                    except Exception:
                        # Parsing failed, so mark the object as absent.
                        missing[oi, tf] = 1
                else:
                    # Missing presence flags or speed means the object is absent.
                    missing[oi, tf] = 1
            except Exception:
                continue

    # Enforce missing = !present. No velocity implies no object.
    missing = (~present.astype(np.bool_)).astype(np.uint8)

    return (torch.from_numpy(vel),
            torch.from_numpy(present.astype(np.bool_)),
            torch.from_numpy(missing.astype(np.bool_)))


def presence_from_seg_masks(seg_prompt_dir: str | Path,
                            label_ids_sorted: Sequence[int]) -> Optional[torch.Tensor]:
    """Build `(O, Tv)` presence booleans from `seg_mask/{video_id}/{prompt_id}/FL2_mask_data/mask_*.npz`.

    Each `.npz` is assumed to contain an `(H, W)` instance-id array where 0 is background.
    """
    base = Path(seg_prompt_dir) / "FL2_mask_data"
    if not base.exists():
        return None
    mask_files = sorted(base.glob("mask_*.npz"))
    if not mask_files:
        return None

    O = len(label_ids_sorted)
    Tv = len(mask_files)
    presence = np.zeros((O, Tv), dtype=np.uint8)
    lid_to_idx = {lid: i for i, lid in enumerate(label_ids_sorted)}

    for t, mf in enumerate(mask_files):
        try:
            with np.load(mf) as data:
                # Use the first key, typically `arr_0`.
                key = list(data.keys())[0]
                mask = data[key]  # (H,W), int
            unique_ids = np.unique(mask)
            # Mark the object present when its label_id appears in this frame.
            for lid in label_ids_sorted:
                if lid != 0 and (unique_ids == lid).any():
                    presence[lid_to_idx[lid], t] = 1
        except Exception:
            continue

    return torch.from_numpy(presence.astype(np.bool_))


def collate_object_inputs(samples: List[SampleObjectInputs]) -> BatchedObjectInputs:
    """
    Pad object count `O` and temporal length `Tv` to the batch maxima.
    """
    if not samples:
        return BatchedObjectInputs(None, None, None, None, None, None, None, None)

    # Integration policy: derive presence from velocity_missing when presence is absent.
    for _s in samples:
        if _s.presence is None and _s.velocity_missing is not None:
            try:
                _s.presence = (~_s.velocity_missing)
            except Exception:
                pass

    # Helper: build coverage mask (True where object considered "present") preferring velocity_missing inverse
    def _coverage_mask(sample: SampleObjectInputs) -> Optional[torch.Tensor]:
        if sample.velocity_missing is not None:
            try:
                return (~sample.velocity_missing)
            except Exception:
                pass
        if sample.presence is not None:
            return sample.presence
        if sample.velocity_f is not None:
            try:
                return sample.velocity_f[..., 0].abs() > 0
            except Exception:
                return None
        return None

    # 1) (Optional) Per-class + Global Hard Top-K (per-sample) based on temporal coverage (now derived from ~velocity_missing primarily)
    #    Config-driven: read from env variable or hydra config injection (lazy import to avoid circular)
    #    We attempt to fetch hydra global config only once; if unavailable, skip.
    try:
        from hydra.core.global_hydra import GlobalHydra  # type: ignore
        gh = GlobalHydra.instance()
        if gh is not None and gh.is_initialized():
            cfg = gh.hydra.runtime.choices  # not actual config; fallback below
        # safer: use HydraConfig to fetch composed config
        from hydra import compose as _compose  # noqa
    except Exception:
        cfg = None  # not used

    # Instead of pulling full hydra config (which can re-compose), inspect a lightweight global variable.
    # Simpler approach: environment variable MMAUDIO_GLOBAL_OBJ_TOPK overrides.
    import os
    env_k = os.environ.get('MMAUDIO_GLOBAL_OBJ_TOPK', None)
    env_pp = os.environ.get('MMAUDIO_PER_PROMPT_TOPK', None)
    global_top_k: int | None = None
    per_prompt_top_k: int | None = None
    if env_k is not None:
        try:
            global_top_k = int(env_k)
        except Exception:
            global_top_k = None
    if env_pp is not None:
        try:
            per_prompt_top_k = int(env_pp)
        except Exception:
            per_prompt_top_k = None
    # Gather original object counts (before filtering/pruning) for logging
    orig_counts: List[int] = []
    def _obj_count(sample: SampleObjectInputs) -> int:
        for attr in ('velocity_f','mass_f','presence','obj_patch_attn'):
            t = getattr(sample, attr)
            if t is not None:
                return t.shape[0]
        return 0
    for _s in samples:
        orig_counts.append(_obj_count(_s))

    # Keep empty samples so batch alignment stays stable even when no objects are present.
    pruned: List[SampleObjectInputs] = []

    for s in samples:
        # Stage A: per-prompt K (if enabled)
        if per_prompt_top_k and per_prompt_top_k > 0:
            presence_mask = _coverage_mask(s)
            if presence_mask is not None:
                try:
                    cov_pp = presence_mask.float().mean(dim=1)
                    O = cov_pp.shape[0]
                    # grouping = prompt_ids (if merged); if single prompt, just global inside sample
                    if s.prompt_ids is not None:
                        pids = s.prompt_ids
                        keep_mask = torch.zeros(O, dtype=torch.bool, device=cov_pp.device)
                        for pid in torch.unique(pids).tolist():
                            idx = (pids == pid).nonzero(as_tuple=True)[0]
                            if idx.numel() <= per_prompt_top_k:
                                keep_mask[idx] = True
                            else:
                                sub_cov = cov_pp[idx]
                                k_sel = min(per_prompt_top_k, sub_cov.shape[0])
                                _, rel_top = torch.topk(sub_cov, k=k_sel)
                                keep_mask[idx[rel_top]] = True
                    else:
                        # single prompt: equivalent to min(per_prompt_top_k, O)
                        k_sel = min(per_prompt_top_k, cov_pp.shape[0])
                        _, top_idx = torch.topk(cov_pp, k=k_sel)
                        keep_mask = torch.zeros_like(cov_pp, dtype=torch.bool); keep_mask[top_idx] = True
                    def _km(t: Optional[torch.Tensor]):
                        return t[keep_mask] if (t is not None) else None
                    s = SampleObjectInputs(
                        velocity_f=_km(s.velocity_f),
                        mass_f=_km(s.mass_f),
                        presence=_km(s.presence),
                        obj_patch_attn=_km(s.obj_patch_attn),
                        label_ids=_km(s.label_ids),
                        prompt_ids=_km(s.prompt_ids),
                        velocity_missing=_km(s.velocity_missing),
                    )
                except Exception:
                    pass
        # Stage B: global K (skip if per-prompt used)
        if (per_prompt_top_k and per_prompt_top_k > 0):
            pruned.append(s)
            continue
        if not global_top_k or global_top_k <= 0:
            pruned.append(s)
            continue
        presence_mask = _coverage_mask(s)
        if presence_mask is None:
            pruned.append(s)
            continue
        try:
            cov = presence_mask.float().mean(dim=1)
        except Exception:
            pruned.append(s)
            continue
        O = cov.shape[0]
        if O <= global_top_k:
            pruned.append(s)
            continue
        _, top_idx = torch.topk(cov, k=global_top_k)
        def _sl(t: Optional[torch.Tensor]):
            return t[top_idx] if (t is not None) else None
        pruned.append(SampleObjectInputs(
            velocity_f=_sl(s.velocity_f),
            mass_f=_sl(s.mass_f),
            presence=_sl(s.presence),
            obj_patch_attn=_sl(s.obj_patch_attn),
            label_ids=_sl(s.label_ids),
            prompt_ids=_sl(s.prompt_ids),
            velocity_missing=_sl(s.velocity_missing),
        ))
    samples = pruned

    # Post-pruning counts + enhanced logging (per-sample before->after pairs)
    pruned_counts: List[int] = []
    for s in samples:
        pruned_counts.append(_obj_count(s))
    empty_mask_tensor: Optional[torch.Tensor] = None
    try:
        if samples:
            total_before = int(sum(orig_counts))
            total_after = int(sum(pruned_counts))
            # mean_before: only non-zero samples to understand typical object load; include zeros separately in pairs
            mean_before = float(np.mean([c for c in orig_counts if c > 0])) if any(c>0 for c in orig_counts) else 0.0
            mean_after = float(np.mean([c for c in pruned_counts if c > 0])) if any(c>0 for c in pruned_counts) else 0.0
            pairs = []
            ratios = []
            for ob, oa in zip(orig_counts, pruned_counts):
                pairs.append(f"{ob}->{oa}")
                r = (oa / ob) if ob > 0 else (0.0 if oa == 0 else 1.0)
                ratios.append(r)
            ratios_str = ",".join(f"{r:.2f}" for r in ratios)
            empty_mask_tensor = torch.tensor([c == 0 for c in pruned_counts], dtype=torch.bool)
            empty_cnt = int(sum(c == 0 for c in pruned_counts))
            # _maybe_log_prune(
            #     f"[object_cond][prune] batch: per_prompt_k={per_prompt_top_k or 0} global_k={global_top_k or 0} "
            #     f"O_before(total={total_before}, mean_nz={mean_before:.2f}) -> O_after(total={total_after}, mean_nz={mean_after:.2f}) "
            #     f"pairs=[{' '.join(pairs)}] ratios=[{ratios_str}] empty={empty_cnt}/{len(pruned_counts)}"
            # )
        else:
            _maybe_log_prune(f"[object_cond][prune] batch EMPTY (no samples)")
    except Exception:
        pass

    B = len(samples)
    Omax = 0
    Tvmax = 0
    Tclip = 0
    GhGw = 0
    has_vel_missing = any(s.velocity_missing is not None for s in samples)
    for s in samples:
        if s.velocity_f is not None:
            Omax = max(Omax, s.velocity_f.shape[0]); Tvmax = max(Tvmax, s.velocity_f.shape[1])
        if s.mass_f is not None:
            Omax = max(Omax, s.mass_f.shape[0])
        if s.presence is not None:
            Omax = max(Omax, s.presence.shape[0]); Tvmax = max(Tvmax, s.presence.shape[1])
        if s.obj_patch_attn is not None:
            Omax = max(Omax, s.obj_patch_attn.shape[0]); Tvmax = max(Tvmax, s.obj_patch_attn.shape[1]); GhGw = max(GhGw, s.obj_patch_attn.shape[2])

    if Omax == 0:
        # All samples empty: still return empty_mask if computed
        return BatchedObjectInputs(None, None, None, None, None, None, empty_mask_tensor, None)

    vel_b = torch.zeros((B, Omax, Tvmax, 1), dtype=torch.float32) if Tvmax > 0 else None
    mass_b = torch.zeros((B, Omax, 1), dtype=torch.float32)
    pres_b = torch.zeros((B, Omax, Tvmax), dtype=torch.bool) if Tvmax > 0 else None
    attn_b = None
    patch_b = torch.zeros((B, Omax, Tvmax, GhGw), dtype=torch.float32) if (Tvmax > 0 and GhGw > 0) else None
    has_label_ids = any(s.label_ids is not None for s in samples)
    has_prompt_ids = any(s.prompt_ids is not None for s in samples)
    # Initialize missing to True for all padded positions; will overwrite real object regions below
    miss_b = torch.ones((B, Omax, Tvmax), dtype=torch.bool) if (Tvmax > 0 and has_vel_missing) else None
    label_b = torch.full((B, Omax), -1, dtype=torch.long) if has_label_ids else None
    prompt_b = torch.full((B, Omax), -1, dtype=torch.long) if has_prompt_ids else None

    for b, s in enumerate(samples):
        if s.velocity_f is not None and vel_b is not None:
            O, Tv = s.velocity_f.shape[:2]
            vel_b[b, :O, :Tv] = s.velocity_f
        if s.mass_f is not None:
            O = s.mass_f.shape[0]
            mass_b[b, :O] = s.mass_f
        # Copy velocity_missing first (source of truth). If absent, fallback to presence inversion.
        if s.velocity_missing is not None and miss_b is not None:
            O, Tv = s.velocity_missing.shape
            miss_b[b, :O, :Tv] = s.velocity_missing
        elif s.presence is not None and miss_b is not None:
            O, Tv = s.presence.shape
            miss_b[b, :O, :Tv] = (~s.presence)
        if s.obj_patch_attn is not None and patch_b is not None:
            O, T, P = s.obj_patch_attn.shape
            patch_b[b, :O, :T, :P] = s.obj_patch_attn
        if label_b is not None and s.label_ids is not None:
            O = s.label_ids.shape[0]
            label_b[b, :O] = s.label_ids
        if prompt_b is not None and s.prompt_ids is not None:
            O = s.prompt_ids.shape[0]
            prompt_b[b, :O] = s.prompt_ids
    if empty_mask_tensor is None:
        empty_mask_tensor = torch.tensor([_obj_count(s) == 0 for s in samples], dtype=torch.bool)

    # Derive presence from velocity_missing (including padding) for consistent relation presence = ~missing
    if miss_b is not None:
        pres_b = (~miss_b)

    return BatchedObjectInputs(vel_b, mass_b, pres_b, patch_b, label_b, prompt_b, empty_mask_tensor, miss_b)


def build_vggsound_val_inputs(
    video_id: str,
    prompt_id: int,
    *,
    mass_root: str | Path,
    velocity_root: str | Path,
    seg_root: str | Path,
    tv: Optional[int] = None,
    patch_grid_hw: Optional[Tuple[int, int]] = None,  # Grid for patch-wise attention (Gh, Gw); if None, will default
    enable_patch_attn: bool = False,
) -> SampleObjectInputs:
    """
    Load object conditions for the VGGSound validation split.
    - mass: {mass_root}/{video_id}/mass_estimation.json
    - velocity: {velocity_root}/{video_id}/{prompt_id}/speeds/instant_speeds.csv
    - seg: {seg_root}/{video_id}/{prompt_id}/FL2_mask_data/mask_*.npz
    - patch_grid_hw: Optional grid size for patch-wise processing (height, width).
    """
    vel_csv = Path(velocity_root) / video_id / str(prompt_id) / "speeds" / "instant_speeds.csv"
    # mass_json = Path(mass_root) / video_id / "mass_estimation.json"
    # seg_dir = Path(seg_root) / video_id / str(prompt_id)
    # Support both the new and legacy layouts.
    mass_json = _resolve_mass_json(mass_root, video_id) or (Path(mass_root) / video_id / "mass_estimation.json")
    seg_dir = _resolve_seg_prompt_dir(seg_root, video_id, prompt_id) or (Path(seg_root) / video_id / str(prompt_id))


    stage_prof: dict[str,float] = {}
    # The velocity CSV is required because it defines the label-id space and temporal length.
    t_stage = time.perf_counter()
    with io_timer("obj_build_velocity"):
        lid2obj = label_map_from_velocity_csv(vel_csv)
        if not lid2obj:
            out_empty = SampleObjectInputs(None, None, None, None)
            setattr(out_empty, 'stage_prof', stage_prof)
            return out_empty
        label_ids_sorted = sorted(lid2obj.keys())
        vel_f, csv_presence, vel_missing = parse_velocity_csv(vel_csv, label_ids_sorted, tv=tv)
    stage_prof['obj_build_velocity'] = stage_prof.get('obj_build_velocity',0.0) + (time.perf_counter()-t_stage)

    # Minimal integration rule: ignore segmentation-derived presence and use `~velocity_missing`.
    presence = (~vel_missing)

    # Mass has no time axis.
    t_stage = time.perf_counter()
    with io_timer("obj_build_mass"):
        mass_f = load_mass_json(mass_json, {lid: lid2obj[lid] for lid in label_ids_sorted})
    stage_prof['obj_build_mass'] = stage_prof.get('obj_build_mass',0.0) + (time.perf_counter()-t_stage)

    obj_patch_attn = None
    if enable_patch_attn:
        try:
            if patch_grid_hw is None:
                patch_grid_hw = (14, 14)
            Gh, Gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
            obj_patch_attn, stage_prof = build_obj_patch_attention(
                seg_dir,
                label_ids_sorted,
                tv=vel_f.shape[1],
                grid_hw=(Gh, Gw),
                stage_prof=stage_prof,
            )
        except Exception:
            obj_patch_attn = None

    # patch_grid_hw is only used to set the grid resolution for building obj_patch_attn.

    sample = SampleObjectInputs(velocity_f=vel_f,
                                mass_f=mass_f,
                                presence=presence,
                                obj_patch_attn=obj_patch_attn,
                                label_ids=torch.tensor(label_ids_sorted, dtype=torch.long),
                                prompt_ids=None,
                                velocity_missing=vel_missing)
    setattr(sample, 'stage_prof', stage_prof)
    return sample


def build_vggsound_val_inputs_all_prompts(
    video_id: str,
    *,
    mass_root: str | Path,
    velocity_root: str | Path,
    seg_root: str | Path,
    tv: Optional[int] = None,
    patch_grid_hw: Optional[Tuple[int, int]] = None,
    enable_patch_attn: bool = False,
) -> SampleObjectInputs:
    """Load ALL prompt_id subdirectories for a video and merge objects.
    - For each prompt_id directory (numeric), parse its instant_speeds.csv similar to build_vggsound_val_inputs.
    - Keep (prompt_id, label_id) pairs distinct; we do NOT attempt to merge identical text across prompts.
    - Return concatenated SampleObjectInputs.
    Per-class Top-K later will group inside each prompt separately (handled via prompt_ids in collate).
    """
    vroot = Path(velocity_root) / video_id
    if not vroot.exists():
        return SampleObjectInputs(None, None, None, None)
    prompt_dirs = [d for d in vroot.iterdir() if d.is_dir() and d.name.isdigit()]
    if not prompt_dirs:
        return SampleObjectInputs(None, None, None, None)
    prompt_dirs.sort(key=lambda p: int(p.name))

    all_vel = []
    all_mass = []
    all_presence = []
    all_patch = []
    all_label_ids = []
    all_prompt_ids = []
    all_vel_missing = []

    agg_stage_prof: dict[str,float] = {}
    for d in prompt_dirs:
        pid = int(d.name)
        try:
            sub = build_vggsound_val_inputs(
                video_id=video_id,
                prompt_id=pid,
                mass_root=mass_root,
                velocity_root=velocity_root,
                seg_root=seg_root,
                tv=tv,
                patch_grid_hw=patch_grid_hw,
                enable_patch_attn=enable_patch_attn,
            )
        except Exception:
            continue
        if sub.velocity_f is None and sub.mass_f is None and sub.presence is None:
            continue
        O_sub = 0
        if sub.velocity_f is not None:
            O_sub = sub.velocity_f.shape[0]
        elif sub.mass_f is not None:
            O_sub = sub.mass_f.shape[0]
        elif sub.presence is not None:
            O_sub = sub.presence.shape[0]
        if O_sub == 0:
            continue
        all_vel.append(sub.velocity_f)
        all_mass.append(sub.mass_f)
        all_presence.append(sub.presence)
        all_patch.append(sub.obj_patch_attn)
        all_vel_missing.append(sub.velocity_missing if sub.velocity_missing is not None else torch.ones((O_sub, sub.velocity_f.shape[1]), dtype=torch.bool))
        # label_ids propagation
        all_label_ids.append(sub.label_ids if sub.label_ids is not None else torch.arange(O_sub))
        all_prompt_ids.append(torch.full((O_sub,), pid, dtype=torch.long))
        # accumulate per-stage profiling (sum over prompts)
        sp = getattr(sub, 'stage_prof', None)
        if sp:
            for k,v in sp.items():
                agg_stage_prof[k] = agg_stage_prof.get(k,0.0)+v

    if not all_label_ids:
        return SampleObjectInputs(None, None, None, None)

    def _cat(lst):
        if any(x is None for x in lst):
            return None
        return torch.cat(lst, dim=0)

    vel_f = _cat(all_vel)
    mass_f = _cat(all_mass)
    presence = _cat(all_presence)
    obj_patch_attn = _cat(all_patch)
    vel_missing = None
    if all_vel_missing and not any(x is None for x in all_vel_missing):
        try:
            vel_missing = torch.cat(all_vel_missing, dim=0)
        except Exception:
            vel_missing = None
    label_ids = torch.cat(all_label_ids, dim=0)
    prompt_ids = torch.cat(all_prompt_ids, dim=0)

    sample = SampleObjectInputs(
        velocity_f=vel_f,
        mass_f=mass_f,
        presence=presence,
        obj_patch_attn=obj_patch_attn,
        label_ids=label_ids,
        prompt_ids=prompt_ids,
        velocity_missing=vel_missing,
    )
    setattr(sample, 'stage_prof', agg_stage_prof)
    return sample


def build_obj_patch_attention(seg_prompt_dir: str | Path,
                              label_ids_sorted: Sequence[int],
                              *,
                              tv: int,
                              grid_hw: Tuple[int, int],
                              stage_prof: Optional[dict[str,float]] = None) -> Tuple[Optional[torch.Tensor], dict[str,float]]:
    """Optimized: build (O,T,Gh*Gw) binary patch attention with caching & vectorization.
    Steps:
      1. Load & cache all frame masks (stack) once (obj_build_seg timer).
      2. Prune labels not present in any of first T frames.
      3. Vectorized block any when H,W divisible by Gh,Gw; fallback to patch slice loop otherwise.
    Preserves original semantics: a patch=1 if any pixel of label appears inside its region.
    """
    if stage_prof is None:
        stage_prof = {}

    seg_prompt_dir = Path(seg_prompt_dir)
    base = seg_prompt_dir / "FL2_mask_data"

    # --------------------------------------------------------------
    # LRU cache for segmentation stacks (across epochs / repeated videos)
    # --------------------------------------------------------------
    global _SEG_STACK_CACHE  # type: ignore
    try:
        _SEG_STACK_CACHE  # noqa
    except NameError:  # first use
        _SEG_STACK_CACHE = OrderedDict()  # type: ignore

    import os
    cache_max = int(os.environ.get("MMAUDIO_SEG_CACHE_SIZE", "32"))  # number of prompt dirs
    limit_to_tv_only = bool(int(os.environ.get("MMAUDIO_SEG_LIMIT_TV", "1")))  # if 1, only load needed first tv frames

    def _evict_if_needed():
        while len(_SEG_STACK_CACHE) > cache_max > 0:
            _SEG_STACK_CACHE.popitem(last=False)

    def _load_stack(base_dir: Path, need_tv: int) -> Optional[np.ndarray]:
        # Return stack (T_src,H,W) with at least need_tv frames (unless fewer exist).
        if not base_dir.exists():
            return None
        key = str(base_dir)
        if key in _SEG_STACK_CACHE:
            stack_cached = _SEG_STACK_CACHE[key]
            # If cache has fewer frames than required AND we might have truncated due to prev tv limit, fall back to reload
            if stack_cached.shape[0] >= need_tv or not limit_to_tv_only:
                # Refresh LRU
                _SEG_STACK_CACHE.move_to_end(key)
                return stack_cached[:need_tv]
        # Not cached or insufficient
        t_io = time.perf_counter()
        with io_timer("obj_build_seg_io"):
            packed_npy = base_dir / "pack_seg_224.npy"  # optional pre-packed file (uint16 or int32)
            stack = None
            if packed_npy.exists():
                try:
                    # mmap to avoid full read (especially large T)
                    stack_all = np.load(packed_npy, mmap_mode='r')  # (T,H,W)
                    stack = stack_all[:need_tv] if limit_to_tv_only else np.array(stack_all)
                except Exception:
                    stack = None
            if stack is None:
                mask_files = sorted(base_dir.glob("mask_*.npz"))
                if not mask_files:
                    return None
                frames = []
                # If limiting, slice list first
                load_files = mask_files[:need_tv] if limit_to_tv_only else mask_files
                for f in load_files:
                    try:
                        with np.load(f) as data:
                            key0 = list(data.keys())[0]
                            frames.append(data[key0])
                    except Exception:
                        continue
                if not frames:
                    return None
                stack = np.stack(frames, axis=0)
        stage_prof['obj_build_seg_io'] = stage_prof.get('obj_build_seg_io',0.0)+(time.perf_counter()-t_io)
        # Insert into cache (store full loaded portion only; if we limited by tv keep truncated stack to save RAM)
        try:
            _SEG_STACK_CACHE[key] = stack if limit_to_tv_only else stack  # identical now, placeholder for future variant
            _SEG_STACK_CACHE.move_to_end(key)
            _evict_if_needed()
        except Exception:
            pass
        return stack

    t_stack = time.perf_counter()
    with io_timer("obj_build_seg_stack"):
        stack = _load_stack(base, tv)
        if stack is None:
            return None, stage_prof
    stage_prof['obj_build_seg_stack'] = stage_prof.get('obj_build_seg_stack',0.0)+(time.perf_counter()-t_stack)

    Gh, Gw = grid_hw
    O_total = len(label_ids_sorted)
    T_src, H, W = stack.shape
    T = min(T_src, int(tv))
    if O_total == 0 or T == 0:
        return None, stage_prof

    # Prefer pre-resized offline file (pack_seg_224.npy). If not used, just slice to T without runtime resize.
    stack = stack[:T]

    out = torch.zeros((O_total, T, Gh * Gw), dtype=torch.float32)

    # # Label pruning (ignore background=0)
    # t_labels = time.perf_counter()
    # with io_timer("obj_build_seg_labels"):
    #     present_lids = np.unique(stack[:T])
    # lid_set = set(int(x) for x in present_lids if x != 0)
    # filtered = [lid for lid in label_ids_sorted if lid in lid_set and lid != 0]
    # stage_prof['obj_build_seg_labels'] = stage_prof.get('obj_build_seg_labels',0.0)+(time.perf_counter()-t_labels)
    # if not filtered:
    #     return out, stage_prof  # all zeros (no object pixels)


    # Label pruning (ignore background=0)
    t_labels = time.perf_counter()
    import os
    disable_prune = True
    if disable_prune:
        # --- PRUNING DISABLED: skip np.unique (fast path) ---
        # present_lids = np.unique(stack[:T])  # <-- disabled
        # lid_set = set(int(x) for x in present_lids if x != 0)
        filtered = [lid for lid in label_ids_sorted if lid != 0]
        stage_prof['obj_build_seg_labels'] = stage_prof.get('obj_build_seg_labels', 0.0) + 0.0
    else:
        with io_timer("obj_build_seg_labels"):
            present_lids = np.unique(stack[:T])
        lid_set = set(int(x) for x in present_lids if x != 0)
        filtered = [lid for lid in label_ids_sorted if lid in lid_set and lid != 0]
        stage_prof['obj_build_seg_labels'] = stage_prof.get('obj_build_seg_labels',0.0)+(time.perf_counter()-t_labels)

    if not filtered:
        return out, stage_prof  # all zeros (no object pixels)



    lid_to_idx = {lid: i for i, lid in enumerate(label_ids_sorted)}
    divisible = (H % Gh == 0 and W % Gw == 0)


    t_attn = time.perf_counter()
    with io_timer("obj_build_attn"):
        if divisible:
            bh, bw = H // Gh, W // Gw
            reshaped = stack[:T].reshape(T, Gh, bh, Gw, bw)
            for lid in filtered:
                m = (reshaped == lid)               # (T,Gh,bh,Gw,bw)
                m_any = m.any(axis=2).any(axis=3)   # (T,Gh,Gw)
                out[lid_to_idx[lid]] = torch.from_numpy(
                    m_any.reshape(T, Gh * Gw).astype(np.float32)
                )
        else:
            # Pre-compute patch slices (approximate partition using rounding)
            stride_h = H / Gh
            stride_w = W / Gw
            patch_slices: list[tuple[int,int,int,int]] = []
            for ph in range(Gh):
                h0 = int(round(ph * stride_h))
                h1 = int(round((ph + 1) * stride_h))
                for pw in range(Gw):
                    w0 = int(round(pw * stride_w))
                    w1 = int(round((pw + 1) * stride_w))
                    patch_slices.append((h0, h1, w0, w1))
            for lid in filtered:
                for t in range(T):
                    frame = stack[t]
                    if lid not in frame:  # quick skip
                        continue
                    for p,(h0,h1,w0,w1) in enumerate(patch_slices):
                        if (frame[h0:h1, w0:w1] == lid).any():
                            out[lid_to_idx[lid], t, p] = 1.0
    stage_prof['obj_build_attn'] = stage_prof.get('obj_build_attn',0.0)+(time.perf_counter()-t_attn)

    return out, stage_prof
