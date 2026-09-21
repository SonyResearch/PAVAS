import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset

import csv
import numpy as np
from .patch_accessor import LazyPatchAccessor
from core.utils.io_profile import ENABLE_IO_PROFILE

from core.utils.dist_utils import local_rank
# Object-condition builder.
from core.data.object_conditions import build_vggsound_val_inputs_all_prompts
import time

log = logging.getLogger()
_condition_gap_counts = defaultdict(int)


def _warn_condition_gap(reason: str, video_id: str, detail: str = '') -> None:
    # Counts are local to each data-loader worker; keep long runs readable.
    _condition_gap_counts[reason] += 1
    count = _condition_gap_counts[reason]
    if count <= 3 or count in (10, 100, 1000) or count % 10000 == 0:
        log.warning(
            'Condition input diagnostic: %s (worker count=%d, video_id=%s)%s',
            reason, count, video_id, f': {detail}' if detail else '',
        )


class ExtractedVGG(Dataset):

    def __init__(
        self,
        tsv_path: Union[str, Path],
        *,
        premade_mmap_dir: Union[str, Path],
        data_dim: dict[str, int],
        # These two arguments are optional and kept for backward compatibility.
        object_cond: dict | None = None,    # {'enabled': bool, 'mass_root': str, 'velocity_root': str, 'seg_root': str, 'force_prompt_id': Optional[int]}
        split_tag: str | None = None,       # e.g. 'train' | 'val' | 'test'
        clip_patches_cfg: dict | None = None,  # {'tsv': str, 'memmap_dir': str} or {'tsv': str, 'npy': str}
        video_root: Union[str, Path, None] = None,
        video_ext: str = '.mp4',
    ):
        super().__init__()

        self.data_dim = data_dim
        self.df_list = pd.read_csv(tsv_path, sep='\t').to_dict('records')
        # ids + optional raw video path root
        self.ids = [d['id'] for d in self.df_list]
        self.video_root = Path(video_root) if video_root is not None else None
        self.video_ext = video_ext

        log.info(f'Loading precomputed mmap from {premade_mmap_dir}')
        # load precomputed memory mapped tensors
        premade_mmap_dir = Path(premade_mmap_dir)
        td = TensorDict.load_memmap(premade_mmap_dir)
        log.info(f'Loaded precomputed mmap from {premade_mmap_dir}')
        self.mean = td['mean']
        self.std = td['std']
        self.clip_features = td['clip_features']
        self.sync_features = td['sync_features']
        self.text_features = td['text_features']

        # Store condition configuration and split.
        self._object_cond = object_cond or {}
        self._split_tag = split_tag or self._object_cond.get('tag', None)

        # Load optional CLIP patch tokens.
        self.clip_patches = None
        self.clip_patches_index = None
        self._clip_patches_id2idx = None
        if clip_patches_cfg is not None:
            try:
                cp_tsv = Path(clip_patches_cfg['tsv'])
                import csv as _csv
                # load id order from tsv
                with cp_tsv.open() as f:
                    reader = _csv.DictReader(f, delimiter='\t')
                    self.clip_patches_index = [row['id'] for row in reader]
                # fast lookup map
                if self.clip_patches_index:
                    self._clip_patches_id2idx = {vid: i for i, vid in enumerate(self.clip_patches_index)}
                # load memmap via TensorDict, or fallback to numpy .npy with mmap
                # Priority: flattened accessor if provided
                if 'flat_dir' in clip_patches_cfg:
                    flat_dir = Path(clip_patches_cfg['flat_dir'])
                    if flat_dir.exists():
                        self.clip_patches = LazyPatchAccessor(flat_dir, cache_items=int(clip_patches_cfg.get('cache_items', 32)))
                # legacy paths
                if self.clip_patches is None:
                    if 'memmap_dir' in clip_patches_cfg:
                        cp_dir = Path(clip_patches_cfg['memmap_dir'])
                        try:
                            from tensordict import TensorDict as _TD
                            td = _TD.load_memmap(cp_dir)
                            if 'clip_patches' in td.keys():
                                self.clip_patches = td['clip_patches']  # (N, T, Gh, Gw, D)
                        except Exception:
                            # Fallback: look for a numpy npy artifact inside the directory
                            cp_npy_fallback = cp_dir / 'clip_patches.npy'
                            if cp_npy_fallback.exists():
                                self.clip_patches = np.load(cp_npy_fallback, mmap_mode='r')
                            else:
                                raise
                    elif 'npy' in clip_patches_cfg:
                        cp_npy = Path(clip_patches_cfg['npy'])
                        self.clip_patches = np.load(cp_npy, mmap_mode='r')  # (N, T, Gh, Gw, D)
                    else:
                        raise KeyError("clip_patches_cfg must include 'memmap_dir' or 'npy' or 'flat_dir'")
            except Exception as e:
                log.warning(f'Failed to load clip_patches from {clip_patches_cfg}: {e}')

        # --- begin patch for sync_features shape ---
        if hasattr(self, "sync_features") and self.sync_features is not None:
            s = self.sync_features
            if s.ndim == 4 and s.shape[1] * s.shape[2] == self.data_dim['sync_seq_len']:
                # [N, 24, 8, 768] -> [N, 192, 768]
                self.sync_features = s.reshape(s.shape[0], s.shape[1]*s.shape[2], s.shape[3])
        # --- end patch ---

        if local_rank == 0:
            log.info(f'Loaded {len(self)} samples.')
            log.info(f'Loaded mean: {self.mean.shape}.')
            log.info(f'Loaded std: {self.std.shape}.')
            log.info(f'Loaded clip_features: {self.clip_features.shape}.')
            log.info(f'Loaded sync_features: {self.sync_features.shape}.')
            log.info(f'Loaded text_features: {self.text_features.shape}.')
            if self.clip_patches is not None:
                try:
                    shape = getattr(self.clip_patches, 'shape', None)
                    if shape is None and isinstance(self.clip_patches, LazyPatchAccessor):
                        # report meta
                        shape = (self.clip_patches.N, self.clip_patches.T, self.clip_patches.Gh, self.clip_patches.Gw, self.clip_patches.D)
                    log.info(f'Loaded clip_patches: {shape}')
                except Exception:
                    pass
                if self.clip_patches_index is not None:
                    log.info(f'clip_patches index entries: {len(self.clip_patches_index)} (may be >= dataset size)')

        assert self.mean.shape[1] == self.data_dim['latent_seq_len'], \
            f'{self.mean.shape[1]} != {self.data_dim["latent_seq_len"]}'
        assert self.std.shape[1] == self.data_dim['latent_seq_len'], \
            f'{self.std.shape[1]} != {self.data_dim["latent_seq_len"]}'

        assert self.clip_features.shape[1] == self.data_dim['clip_seq_len'], \
            f'{self.clip_features.shape[1]} != {self.data_dim["clip_seq_len"]}'
        assert self.sync_features.shape[1] == self.data_dim['sync_seq_len'], \
            f'{self.sync_features.shape[1]} != {self.data_dim["sync_seq_len"]}'
        assert self.text_features.shape[1] == self.data_dim['text_seq_len'], \
            f'{self.text_features.shape[1]} != {self.data_dim["text_seq_len"]}'

        assert self.clip_features.shape[-1] == self.data_dim['clip_dim'], \
            f'{self.clip_features.shape[-1]} != {self.data_dim["clip_dim"]}'
        assert self.sync_features.shape[-1] == self.data_dim['sync_dim'], \
            f'{self.sync_features.shape[-1]} != {self.data_dim["sync_dim"]}'
        assert self.text_features.shape[-1] == self.data_dim['text_dim'], \
            f'{self.text_features.shape[-1]} != {self.data_dim["text_dim"]}'

        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.mean
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def get_memory_mapped_tensor(self) -> TensorDict:
        td = TensorDict({
            'mean': self.mean,
            'std': self.std,
            'clip_features': self.clip_features,
            'sync_features': self.sync_features,
            'text_features': self.text_features,
        })
        return td

    # Helper: infer prompt_id by matching `text_prompt` in the velocity CSV against the caption.
    def _guess_prompt_id(self, video_id: str, caption: str) -> int:
        oc = self._object_cond
        vel_root = oc.get('velocity_root', None)
        if vel_root is None:
            return 0
        vdir = Path(vel_root) / video_id
        if not vdir.exists():
            return 0

        # Priority 1: use force_prompt_id when provided.
        if 'force_prompt_id' in oc and isinstance(oc['force_prompt_id'], int):
            return int(oc['force_prompt_id'])

        # List candidate prompt-id directories.
        cand = [p for p in vdir.iterdir() if p.is_dir() and p.name.isdigit()]
        cand.sort(key=lambda p: int(p.name))

        # Prefer CSV files whose text_prompt matches the caption.
        for d in cand:
            csv_path = d / 'speeds' / 'instant_speeds.csv'
            if not csv_path.exists():
                continue
            try:
                with csv_path.open(newline='') as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        tp = r.get('text_prompt', None)
                        if tp is not None and str(tp).strip() == str(caption).strip():
                            return int(d.name)
            except Exception:
                continue

        # Fallback: use the first valid directory.
        return int(cand[0].name) if cand else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = {
            'id': self.df_list[idx]['id'],
            'a_mean': self.mean[idx],
            'a_std': self.std[idx],
            'clip_features': self.clip_features[idx],
            'sync_features': self.sync_features[idx],
            'text_features': self.text_features[idx],
            'caption': self.df_list[idx]['label'],
            'video_exist': self.video_exist,
            'text_exist': self.text_exist,
        }

        # Attach raw video path if available
        if self.video_root is not None:
            try:
                vid = str(data['id'])
                vpath = self.video_root / f'{vid}{self.video_ext}'
                if vpath.exists():
                    data['video_path'] = str(vpath)
            except Exception:
                pass

        # Attach CLIP patches by video id when present.
        prof = {}
        try:
            if self.clip_patches is not None and self._clip_patches_id2idx is not None:
                vid = str(data['id'])
                j = self._clip_patches_id2idx.get(vid)
                if j is not None:
                    t0 = time.perf_counter() if ENABLE_IO_PROFILE else None
                    cp_item = self.clip_patches[j]
                    if t0 is not None:
                        dt = time.perf_counter()-t0
                        # Heuristic: treat dt<0.0008s as hit, else miss (rough threshold adjustable)
                        prof_key = 'patch_hit_time' if dt < 0.0008 else 'patch_miss_time'
                        prof[prof_key] = prof.get(prof_key,0.0)+dt
                    data['clip_patches'] = cp_item
                else:
                    _warn_condition_gap('patch_id_missing', vid)
        except Exception as e:
            _warn_condition_gap('patch_read_failed', str(data['id']), str(e))

        # Load object conditions for both validation and training.
        try:
            oc = self._object_cond
            enabled = bool(oc.get('enabled', False))
            if enabled:
                video_id = str(data['id'])  # TSV id is also the video_id, e.g. "_xxxx_000123".
                caption = data['caption']
                prompt_id = self._guess_prompt_id(video_id, caption)
                patch_grid_hw = None
                tv = None
                
                cp = getattr(self, 'clip_patches', None)
                if cp is not None:
                    try:
                        if isinstance(cp, LazyPatchAccessor):
                            tv = int(cp.T)
                            patch_grid_hw = (int(cp.Gh), int(cp.Gw))
                        else:
                            shp = getattr(cp, 'shape', None)
                            if shp is not None:
                                if len(shp) == 5:
                                    _, Tv_p, Gh, Gw, _ = shp
                                    tv = int(Tv_p)
                                    patch_grid_hw = (int(Gh), int(Gw))
                                elif len(shp) == 4:
                                    # Flattened case: (N, T, S, D).
                                    _, Tv_p, S, _ = shp
                                    tv = int(Tv_p)
                                    # Check whether S == 14*14=196; otherwise infer the grid via sqrt.
                                    if S == 196:
                                        patch_grid_hw = (14, 14)
                                    else:
                                        g = int(S ** 0.5)
                                        if g * g == S:
                                            patch_grid_hw = (g, g)
                        if tv is None:
                            tv = 64  # fallback
                    except Exception:
                        # Force the fallback path.
                        if tv is None:
                            tv = 64
                            patch_grid_hw = (14, 14)


                # Always merge all prompts (multi-prompt default)
                t_obj0 = time.perf_counter() if ENABLE_IO_PROFILE else None
                obj = build_vggsound_val_inputs_all_prompts(
                    video_id=video_id,
                    mass_root=oc['mass_root'],
                    velocity_root=oc['velocity_root'],
                    seg_root=oc['seg_root'],
                    tv=tv,
                    patch_grid_hw=patch_grid_hw,
                    enable_patch_attn=bool(oc.get('enable_patch_attn', False)),
                )
                if all(getattr(obj, key) is None for key in ('velocity_f', 'mass_f', 'presence', 'obj_patch_attn')):
                    _warn_condition_gap('object_inputs_absent_or_no_moving', video_id)
                if t_obj0 is not None:
                    prof['obj_build_time'] = prof.get('obj_build_time',0.0)+(time.perf_counter()-t_obj0)
                if obj.velocity_f is not None:
                    data['velocity_f'] = obj.velocity_f
                if obj.mass_f is not None:
                    data['mass_f'] = obj.mass_f
                if obj.presence is not None:
                    data['presence_mask'] = obj.presence
                if obj.obj_patch_attn is not None:
                    data['obj_patch_attn'] = obj.obj_patch_attn
                if getattr(obj, 'label_ids', None) is not None:
                    data['obj_label_ids'] = obj.label_ids
                if getattr(obj, 'prompt_ids', None) is not None:
                    data['obj_prompt_ids'] = obj.prompt_ids
                if getattr(obj, 'velocity_missing', None) is not None:
                    data['velocity_missing'] = obj.velocity_missing
                # propagate fine-grained stage profiling into per-sample _prof so mm_dataset aggregation catches it
                sp = getattr(obj, 'stage_prof', None)
                if ENABLE_IO_PROFILE and sp:
                    for k,v in sp.items():
                        prof[k] = prof.get(k, 0.0) + v
        except Exception as e:
            _warn_condition_gap('object_inputs_load_failed', str(data['id']), str(e))

        if ENABLE_IO_PROFILE and prof:
            data['_prof'] = prof
        return data

    def __len__(self):
        return len(self.ids)
