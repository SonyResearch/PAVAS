import bisect
import logging
from functools import reduce
from typing import Dict, Iterable, List

import torch
from core.utils.io_profile import io_timer, ENABLE_IO_PROFILE
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from core.data.object_conditions import SampleObjectInputs, collate_object_inputs


log = logging.getLogger(__name__)
_mixed_patch_batches = 0


# modified from https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
class MultiModalDataset(Dataset):
    datasets: list[Dataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, video_datasets: list[Dataset], audio_datasets: list[Dataset]):
        super().__init__()
        self.video_datasets = list(video_datasets)
        self.audio_datasets = list(audio_datasets)
        self.datasets = self.video_datasets + self.audio_datasets

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_datasets[0].compute_latent_stats()


def _intersection_keys(batch: List[Dict]) -> Iterable[str]:
    if not batch:
        return []
    ks = [set(b.keys()) for b in batch]
    inter = reduce(lambda a, b: a & b, ks)
    return list(inter)


def collate_fn(batch: List[Dict]):
    """
    - Merge shared keys with `default_collate`.
    - Pad and merge object-condition tensors separately.
    - Ignore non-shared keys to keep mixed-modality batches safe.
    """
    if not batch:
        return {}

    global _mixed_patch_batches
    patch_count = sum('clip_patches' in item for item in batch)
    if 0 < patch_count < len(batch):
        _mixed_patch_batches += 1
        if _mixed_patch_batches <= 3 or _mixed_patch_batches in (10, 100, 1000) or _mixed_patch_batches % 10000 == 0:
            log.warning(
                'Condition input diagnostic: mixed_patch_batch (worker count=%d, present=%d/%d); '
                'the existing shared-key collation omits clip_patches for this batch',
                _mixed_patch_batches, patch_count, len(batch),
            )

    cond_keys = {'velocity_f', 'mass_f', 'presence_mask', 'obj_patch_attn', 'obj_label_ids', 'obj_prompt_ids', 'obj_empty_mask', 'velocity_missing'}
    common_keys = [k for k in _intersection_keys(batch) if k not in cond_keys]

    collated: Dict = {}
    for k in common_keys:
        with io_timer('collate_field'):
            try:
                collated[k] = default_collate([b[k] for b in batch])
            except Exception:
                collated[k] = [b[k] for b in batch]

    # Merge object conditions separately from the common fields.
    obj_samples: List[SampleObjectInputs] = []
    prof_dicts = []
    for b in batch:
        if '_prof' in b:
            prof_dicts.append(b['_prof'])
        # Derive presence_mask from velocity_missing when it is absent.
        if 'presence_mask' not in b and 'velocity_missing' in b and b.get('velocity_missing') is not None:
            try:
                b['presence_mask'] = (~b['velocity_missing'])
            except Exception:
                pass
        obj_samples.append(SampleObjectInputs(
            velocity_f=b.get('velocity_f'),
            mass_f=b.get('mass_f'),
            presence=b.get('presence_mask'),
            obj_patch_attn=b.get('obj_patch_attn'),
            label_ids=b.get('obj_label_ids'),
            prompt_ids=b.get('obj_prompt_ids'),
            velocity_missing=b.get('velocity_missing'),
        ))

    if any((s.velocity_f is not None or s.mass_f is not None or s.presence is not None or s.obj_patch_attn is not None)
           for s in obj_samples):
        with io_timer('collate_object_inputs'):
            packed = collate_object_inputs(obj_samples)
        if packed.velocity_f is not None:
            collated['velocity_f'] = packed.velocity_f
        if packed.mass_f is not None:
            collated['mass_f'] = packed.mass_f
        if packed.presence is not None:
            collated['presence_mask'] = packed.presence
        if packed.obj_patch_attn is not None:
            collated['obj_patch_attn'] = packed.obj_patch_attn
        if packed.label_ids is not None:
            collated['obj_label_ids'] = packed.label_ids
        if packed.prompt_ids is not None:
            collated['obj_prompt_ids'] = packed.prompt_ids
        if packed.empty_mask is not None:
            collated['obj_empty_mask'] = packed.empty_mask
        if packed.velocity_missing is not None:
            collated['velocity_missing'] = packed.velocity_missing

    # Aggregate per-sample profiling metrics (produced inside worker processes)
    if prof_dicts:
        agg: dict[str, float] = {}
        for d in prof_dicts:
            for k,v in d.items():
                if isinstance(v,(int,float)):
                    agg[k] = agg.get(k,0.0)+float(v)
        if agg:
            collated['__io_prof'] = agg

    return collated
