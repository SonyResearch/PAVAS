import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from core.utils.io_profile import io_timer, ENABLE_IO_PROFILE


class LazyPatchAccessor:
    """Lazy accessor for flattened CLIP patch tokens.

    Expects directory structure:
        flat_dir/
            clip_patches_fp16_flat.npy   (N, T, S=Gh*Gw, D) float16/float32
            meta_clip_patches.json       {N,T,Gh,Gw,D,...}

    Returns tensors shaped (T, Gh, Gw, D) on demand.
    Maintains a tiny LRU cache to avoid repeated numpy->torch wrapping.
    Tracks latency + hit/miss stats for diagnostics.
    """

    def __init__(
        self,
        flat_dir: str | Path,
        *,
        cache_items: int = 32,
        to_float32: bool = False,
        log_interval: int = 500,
        warmup: int = 128,
    ):
        self.flat_dir = Path(flat_dir)
        meta_path = self.flat_dir / 'meta_clip_patches.json'
        arr_path = self.flat_dir / 'clip_patches_fp16_flat.npy'
        if not meta_path.exists():
            raise FileNotFoundError(f"meta file missing: {meta_path}")
        if not arr_path.exists():
            raise FileNotFoundError(f"array file missing: {arr_path}")
        with meta_path.open() as f:
            self.meta = json.load(f)
        self._np_arr = np.load(arr_path, mmap_mode='r')  # (N,T,S,D)
        self.N = int(self.meta['N'])
        self.T = int(self.meta['T'])
        self.Gh = int(self.meta['Gh'])
        self.Gw = int(self.meta['Gw'])
        self.D = int(self.meta['D'])
        self.S = self.Gh * self.Gw
        assert self._np_arr.shape[:3] == (self.N, self.T, self.S), (
            f"shape mismatch array={self._np_arr.shape} meta=(N={self.N},T={self.T},S={self.S})")

        self.cache_items = cache_items
        self.to_float32 = to_float32
        self.log_interval = log_interval
        self.warmup = warmup

        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        # stats
        self._hits = 0
        self._misses = 0
        self._loads_time_sum = 0.0
        self._loads_time_sum_sq = 0.0
        self._accesses = 0
        self._first_ts = time.time()

    @property
    def grid_hw(self):  # (Gh, Gw)
        return self.Gh, self.Gw

    def _record_latency(self, dt: float, hit: bool):
        if hit:
            self._hits += 1
        else:
            self._misses += 1
            self._loads_time_sum += dt
            self._loads_time_sum_sq += dt * dt
        self._accesses += 1
        if self.log_interval and (self._accesses % self.log_interval == 0):
            self._log_stats(prefix='[LazyPatchAccessor][periodic]')

    def _log_stats(self, prefix='[LazyPatchAccessor][stats]'):
        total = max(1, self._hits + self._misses)
        miss = self._misses
        miss_rate = miss / total
        avg = self._loads_time_sum / max(1, miss)
        if miss > 1:
            var = (self._loads_time_sum_sq / miss) - (avg * avg)
            std = var ** 0.5 if var > 0 else 0.0
        else:
            std = 0.0
        warm = min(self.warmup, self._misses)
        print(f"{prefix} accesses={self._accesses} hit={self._hits} miss={self._misses} miss_rate={miss_rate*100:.2f}% avg_load={avg*1000:.2f}ms std={std*1000:.2f}ms warmup_miss={warm}")

    def report_stats(self, reset=False):
        self._log_stats()
        if reset:
            self._hits = self._misses = 0
            self._loads_time_sum = self._loads_time_sum_sq = 0.0
            self._accesses = 0
            self._first_ts = time.time()

    def __len__(self):
        return self.N

    def _evict_if_needed(self):
        if len(self._cache) > self.cache_items:
            # pop oldest
            self._cache.pop(next(iter(self._cache)))

    def _fetch(self, idx: int) -> torch.Tensor:
        # numpy slice (T,S,D)
        np_view = self._np_arr[idx]  # still memmap-backed
        # create torch view (float16)
        tensor = torch.from_numpy(np_view)  # (T,S,D)
        if self.to_float32 and tensor.dtype == torch.float16:
            tensor = tensor.float()
        # reshape to (T,Gh,Gw,D)
        tensor = tensor.view(self.T, self.Gh, self.Gw, self.D)
        return tensor

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.N:
            raise IndexError(idx)
        t0 = time.perf_counter()
        cached = self._cache.get(idx, None)
        if cached is not None:
            self._cache.move_to_end(idx)
            self._record_latency(time.perf_counter() - t0, hit=True)
            return cached
        with io_timer('patch_fetch_miss'):
            tensor = self._fetch(idx)
        self._cache[idx] = tensor
        self._evict_if_needed()
        self._record_latency(time.perf_counter() - t0, hit=False)
        return tensor


__all__ = [
    'LazyPatchAccessor'
]
