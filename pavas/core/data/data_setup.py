import logging
import random
import torch.distributed as dist
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from core.data.eval.audiocaps import AudioCapsData
from core.data.eval.video_dataset import MovieGen, VGGSound
from core.data.extracted_audio import ExtractedAudio
from core.data.extracted_vgg import ExtractedVGG
from core.data.mm_dataset import MultiModalDataset
from core.data.mm_dataset import collate_fn as mm_collate_fn
from core.utils.dist_utils import local_rank

from pathlib import Path
from functools import lru_cache
import glob

log = logging.getLogger()

# Mass / segmentation path resolvers supporting both the old and new layouts.
@lru_cache(maxsize=200_000)
def _resolve_mass_path(mass_root: str, vid: str) -> Optional[Path]:
    root = Path(mass_root)
    direct = root / vid / "mass_estimation.json"
    if direct.exists():
        return direct
    hits = list(root.glob(f"split*_json/{vid}/mass_estimation.json"))
    return hits[0] if hits else None

@lru_cache(maxsize=200_000)
def _resolve_seg_dir(seg_root: str, vid: str, track_id: str | int) -> Optional[Path]:
    root = Path(seg_root)
    tid = str(track_id)
    direct = root / vid / tid / "FL2_mask_data"
    if direct.is_dir():
        return direct
    hits = list(root.glob(f"split_*/{vid}/{tid}/FL2_mask_data"))
    return hits[0] if hits else None



# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}')


def load_vgg_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    """Construct an `ExtractedVGG` dataset.

    Pass `object_cond` and `tag` to enable object-condition loading for train/val splits.
    """
    dataset = ExtractedVGG(
        tsv_path=data_cfg.tsv,
        data_dim=cfg.data_dim,
        premade_mmap_dir=data_cfg.memmap_dir,
        object_cond=getattr(data_cfg, 'object_cond', None),
        split_tag=getattr(data_cfg, 'tag', None),       # e.g. 'train' | 'val' | 'test'
        clip_patches_cfg=getattr(data_cfg, 'clip_patches', None),  # optional: {tsv: str, memmap: str}
        video_root=getattr(cfg, 'video_root', None),
        video_ext=getattr(cfg, 'video_ext', '.mp4'),
    )
    return dataset


def load_audio_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    dataset = ExtractedAudio(tsv_path=data_cfg.tsv,
                             data_dim=cfg.data_dim,
                             premade_mmap_dir=data_cfg.memmap_dir)
    return dataset


def setup_training_datasets(cfg: DictConfig) -> tuple[Dataset, DistributedSampler, DataLoader]:
    """Build the training dataset configuration.

    When `finetune.enabled=True`, use only VGG and choose the split from `cfg.finetune.split`.
    Otherwise, follow the standard multimodal training setup.
    """
    # finetune config lives under cfg.data.finetune (since defaults include data: base)
    finetune_cfg = getattr(getattr(cfg, 'data', {}), 'finetune', None)
    if finetune_cfg and getattr(finetune_cfg, 'enabled', False):
        split = getattr(finetune_cfg, 'split', 'val')
        if split == 'val':
            vgg_cfg = cfg.data.ExtractedVGG_val
        elif split == 'train':
            vgg_cfg = cfg.data.ExtractedVGG
        else:
            raise ValueError(f'finetune.split must be "val" or "train", got {split}')
        dataset = load_vgg_data(cfg, vgg_cfg)
    elif getattr(cfg, 'mini_train', False):
        vgg = load_vgg_data(cfg, cfg.data.ExtractedVGG_val)
        audiocaps = load_audio_data(cfg, cfg.data.AudioCaps)
        dataset = MultiModalDataset([vgg], [audiocaps])
    elif getattr(cfg, 'example_train', False):
        video = load_vgg_data(cfg, cfg.data.Example_video)
        audio = load_audio_data(cfg, cfg.data.Example_audio)
        dataset = MultiModalDataset([video], [audio])
    else:
        # load the largest one first
        freesound = load_audio_data(cfg, cfg.data.FreeSound)
        vgg = load_vgg_data(cfg, cfg.data.ExtractedVGG)
        audiocaps = load_audio_data(cfg, cfg.data.AudioCaps)
        audioset_sl = load_audio_data(cfg, cfg.data.AudioSetSL)
        bbcsound = load_audio_data(cfg, cfg.data.BBCSound)
        clotho = load_audio_data(cfg, cfg.data.Clotho)
        dataset = MultiModalDataset([vgg] * cfg.vgg_oversample_rate,
                                    [audiocaps, audioset_sl, bbcsound, freesound, clotho])

    # Optionally filter samples without prompt directories.

    def _maybe_filter_prompts(ds: Dataset) -> Dataset:
        try:
            from core.data.extracted_vgg import ExtractedVGG  # local import to avoid cycles
            if isinstance(ds, ExtractedVGG):
                oc = getattr(ds, '_object_cond', {}) or {}
                if not oc.get('enabled', False):
                    return ds
                if not oc.get('skip_no_prompt', False):
                    return ds
                vel_root = oc.get('velocity_root', None)
                mass_root = oc.get('mass_root', None)
                if vel_root is None or mass_root is None:
                    return ds
                from pathlib import Path
                vel_root = Path(vel_root)
                keep_indices = []
                skipped: list[str] = []
                for i, row in enumerate(ds.df_list):
                    vid = str(row['id'])
                    vdir = vel_root / vid
                    # Check both the new and legacy layouts.
                    mass_file = _resolve_mass_path(mass_root, vid)

                    # Condition 1: the mass file exists.
                    if mass_file is None or not mass_file.exists():
                        skipped.append(f'{vid}:no_mass')
                        continue
                    # Condition 2: at least one prompt directory contains a CSV file.
                    ok_prompt = False
                    if vdir.exists():
                        try:
                            for sub in vdir.iterdir():
                                if sub.is_dir() and sub.name.isdigit():
                                    csv_path = sub / 'speeds' / 'instant_speeds.csv'
                                    if csv_path.exists():
                                        ok_prompt = True
                                        break
                        except Exception:
                            ok_prompt = False
                    if not ok_prompt:
                        skipped.append(f'{vid}:no_prompt')
                        continue
                    keep_indices.append(i)
                if not keep_indices:
                    log.warning('[prompt-skip] All samples filtered out; returning original dataset to avoid empty training set.')
                    if skipped:
                        preview = ', '.join(skipped[:50])
                        log.warning(f'[prompt-skip] Skipped videos ({len(skipped)} total): {preview}')
                    return ds
                if len(keep_indices) != len(ds):
                    log.info(f'[prompt-skip] Keeping {len(keep_indices)}/{len(ds)} samples (mass+prompt ok). Skipped {len(skipped)}.')
                    if skipped:
                        preview = ', '.join(skipped[:50])
                        log.info(f'[prompt-skip] Example skipped: {preview}')
                subset = Subset(ds, keep_indices)
                if hasattr(ds, 'compute_latent_stats') and not hasattr(subset, 'compute_latent_stats'):
                    subset.compute_latent_stats = ds.compute_latent_stats  # type: ignore[attr-defined]
                return subset
        except Exception as e:
            log.warning(f'Prompt skip filtering failed: {e}')
        return ds

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory

    # Use mm_collate_fn whenever VGG-derived object conditions may be present.
    use_mm_collate = isinstance(dataset, ExtractedVGG) or (
        isinstance(dataset, MultiModalDataset) and any(isinstance(d, ExtractedVGG) for d in dataset.video_datasets)
    )
    # Apply filtering for single ExtractedVGG cases only (not MultiModal oversampling variants)
    if isinstance(dataset, ExtractedVGG):
        dataset = _maybe_filter_prompts(dataset)

    sampler, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        collate_fn=mm_collate_fn if use_mm_collate else None,
    )

    return dataset, sampler, loader


def setup_test_datasets(cfg):
    dataset = load_vgg_data(cfg, cfg.data.ExtractedVGG_test)

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=mm_collate_fn
    )

    return dataset, sampler, loader



def setup_val_datasets(cfg: DictConfig) -> tuple[Dataset, DataLoader, DataLoader]:
    if getattr(cfg, 'example_train', False):
        dataset = load_vgg_data(cfg, cfg.data.Example_video)
    else:
        dataset = load_vgg_data(cfg, cfg.data.ExtractedVGG_val)

    val_batch_size = cfg.batch_size
    val_eval_batch_size = cfg.eval_batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, val_loader = construct_loader(
        dataset,
        val_batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=mm_collate_fn
    )
    _, eval_loader = construct_loader(
        dataset,
        val_eval_batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=mm_collate_fn
    )

    return dataset, val_loader, eval_loader


def setup_eval_dataset(dataset_name: str, cfg: DictConfig) -> tuple[Dataset, DataLoader]:
    if dataset_name.startswith('audiocaps_full'):
        dataset = AudioCapsData(cfg.eval_data.AudioCaps_full.audio_path,
                                cfg.eval_data.AudioCaps_full.csv_path)
    elif dataset_name.startswith('audiocaps'):
        dataset = AudioCapsData(cfg.eval_data.AudioCaps.audio_path,
                                cfg.eval_data.AudioCaps.csv_path)
    elif dataset_name.startswith('moviegen'):
        dataset = MovieGen(cfg.eval_data.MovieGen.video_path,
                           cfg.eval_data.MovieGen.jsonl_path,
                           duration_sec=cfg.duration_s)
    elif dataset_name.startswith('vggsound'):
        dataset = VGGSound(cfg.eval_data.VGGSound.video_path,
                           cfg.eval_data.VGGSound.csv_path,
                           duration_sec=cfg.duration_s)
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, loader = construct_loader(dataset,
                                 batch_size,
                                 num_workers,
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=pin_memory,
                                 error_avoidance=True)
    return dataset, loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def construct_loader(dataset: Dataset,
                     batch_size: int,
                     num_workers: int,
                     *,
                     shuffle: bool = True,
                     drop_last: bool = True,
                     pin_memory: bool = False,
                     error_avoidance: bool = False,
                     collate_fn: Optional[Callable] = None) -> tuple[DistributedSampler, DataLoader]:
    # Use DistributedSampler only when the process group is initialized.
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
        train_loader = DataLoader(dataset,
                                  batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=drop_last,
                                  persistent_workers=num_workers > 0,
                                  pin_memory=pin_memory,
                                  # Priority: explicit collate_fn > error_avoidance_collate > default
                                  collate_fn=(collate_fn if collate_fn is not None
                                              else (error_avoidance_collate if error_avoidance else None)))
    else:
        # Single-process / debug mode: do not use DistributedSampler
        train_sampler = None
        train_loader = DataLoader(dataset,
                                  batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init_fn,
                                  drop_last=drop_last,
                                  persistent_workers=num_workers > 0,
                                  pin_memory=pin_memory,
                                  collate_fn=(collate_fn if collate_fn is not None
                                              else (error_avoidance_collate if error_avoidance else None)))

    return train_sampler, train_loader
