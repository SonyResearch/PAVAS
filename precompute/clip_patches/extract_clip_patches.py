import logging
import os
import sys
import tempfile
from pathlib import Path

# Keep the core package importable when this script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "pavas"
CACHE_ROOT = Path(tempfile.gettempdir()) / 'pavas-cache'
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_ROOT / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(CACHE_ROOT / 'numba'))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Set environment variables before any other imports.
os.environ["TORCHVISION_VIDEO_BACKEND"] = "pyav"

from argparse import ArgumentParser
from datetime import timedelta

import pandas as pd
import tensordict as td
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

from core.data.extraction.vgg_sound_clip_patches import VGGSoundClipPatches
from core.model.utils.features_utils import FeaturesUtils
from core.utils.dist_utils import local_rank, world_size

import torchvision
import av

# Defaults matching 8s / 8 FPS video chunks used elsewhere
_DURATION_SEC = 8.0
_CLIP_FPS = 8.0

log = logging.getLogger()
log.setLevel(logging.INFO)

# Work around system cuDNN symbol mismatch by disabling cuDNN usage.
# Patch extraction mainly uses ViT (Conv2d for patch embed + MHA), which runs fine without cuDNN.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Also force scaled-dot-product attention to math kernels to avoid cuDNN/Flash paths
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass


def distributed_setup():
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    # Select a safe CUDA device when GPUs are available.
    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(local_rank % device_count)
        log.info(f"Set CUDA device to {local_rank % device_count}")
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}, device_count={device_count}")


def _safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # simple dict-of-lists to list-of-dicts merge assuming uniform keys
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        # tensors stack, lists keep, others as-is
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.utils.data._utils.collate.default_collate(vals)
        else:
            out[k] = vals
    return out


def setup_loader(video_root: Path, tsv_path: Path, batch_size: int, num_workers: int, clip_size: int):
    dataset = VGGSoundClipPatches(video_root,
                                  tsv_path=tsv_path,
                                  duration_sec=_DURATION_SEC,
                                  clip_fps=_CLIP_FPS,
                                  clip_size=clip_size)
    sampler = DistributedSampler(dataset, rank=local_rank, shuffle=False)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=sampler,
                        drop_last=False,
                        collate_fn=_safe_collate,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return dataset, loader


@torch.inference_mode()
def main():
    parser = ArgumentParser()
    parser.add_argument("--video_root", type=Path, required=True,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--tsv", type=Path, required=True,
                        help="TSV with id and label columns")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to write memmap + TSV")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=_CLIP_FPS,
                        help="Target FPS for CLIP stream; VGGSound loader uses 8 by default")
    parser.add_argument("--clip_model", type=str, choices=["vit_b_16","vit_l_14","vit_h_14_384","dinov2_s_14"], default="vit_b_16",
                        help="Which vision backbone to use for patch extraction (CLIP or DINOv2).")
    parser.add_argument("--clip_size", type=int, default=224,
                        help="Spatial size given to CLIP (e.g., 224/336/384). Smaller reduces patch grid.")
    parser.add_argument("--grid_out", type=int, default=0,
                        help="If >0, bilinear downsample spatial grid to grid_out x grid_out before saving.")
    parser.add_argument("--temporal_stride", type=int, default=1,
                        help="Subsample frames by this stride (e.g., 4 makes T=16 from 64).")
    parser.add_argument("--merge_only", action="store_true",
                        help="Skip extraction and only merge existing shards under output_dir/patch_shards")
    parser.add_argument("--direct_memmap", action="store_true",
                        help="Write features directly into a preallocated memmap (no shards, no merge). Greatly faster.")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16",
                        help="Output dtype to store in memmap (default: float16)")
    args = parser.parse_args()

    # Fast single-process merge path: skip distributed/NCCL entirely for --merge_only
    if args.merge_only:
        log.info("[merge_only] Start (no distributed init)")
        base = args.output_dir / "patch_shards"
        if not base.exists():
            log.error(f"[merge_only] Shard directory not found: {base}")
            return
        shard_paths = []
        meta_paths = []
        total_bytes = 0
        for rank_name in sorted(os.listdir(base)):
            rank_dir = base / rank_name
            for shard_name in sorted(os.listdir(rank_dir)):
                sp = rank_dir / shard_name
                if sp.suffix == '.pth':
                    shard_paths.append(str(sp))
                    try:
                        total_bytes += os.path.getsize(sp)
                    except Exception:
                        pass
                elif shard_name.endswith('.meta.tsv'):
                    meta_paths.append(str(sp))
        log.info(f"[merge_only] Found {len(shard_paths)} shards (+{len(meta_paths)} metas), approx size {total_bytes/1e9:.1f} GB (raw)")

        # First pass: ids/labels + shape/dtype, prefer metadata
        used_id = set(); ids = []; labels = []
        sample_shape = None; sample_dtype = None
        if len(meta_paths) > 0:
            for mp in tqdm(meta_paths, desc="Scanning shard metadata", leave=False):
                mdf = pd.read_csv(mp, sep='\t')
                if sample_shape is None and {'T','Gh','Gw','D','dtype'}.issubset(mdf.columns):
                    sample_shape = (int(mdf['T'][0]), int(mdf['Gh'][0]), int(mdf['Gw'][0]), int(mdf['D'][0]))
                    sample_dtype = str(mdf['dtype'][0])
                for _, row in mdf.iterrows():
                    sid = row['id']
                    if sid in used_id: continue
                    used_id.add(sid); ids.append(sid); labels.append(row['label'])
        else:
            for sp in tqdm(shard_paths, desc="Scanning shards", leave=False):
                sh = torch.load(sp, map_location='cpu')
                clips = sh['clip_patches']
                if sample_shape is None:
                    sample_shape = tuple(clips.shape[1:])
                    sample_dtype = str(clips.dtype).replace('torch.', '')
                for i, sid in enumerate(sh['id']):
                    if sid in used_id: continue
                    used_id.add(sid); ids.append(sid); labels.append(sh['caption'][i])
        N = len(ids)
        if N == 0:
            log.warning("[merge_only] No samples found to merge.")
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'id': ids, 'label': labels}).to_csv(
            args.output_dir / 'vgg-patches.tsv', sep='\t', index=False)
        T, Gh, Gw, D = sample_shape
        dtype_np = np.float16 if 'float16' in sample_dtype else np.float32
        out_dir = args.output_dir / 'vgg-patches'
        out_dir.mkdir(parents=True, exist_ok=True)
        mmap_path = out_dir / 'clip_patches.npy'
        mm = np.lib.format.open_memmap(mmap_path, mode='w+', dtype=dtype_np, shape=(N, T, Gh, Gw, D))
        id_to_idx = {sid: i for i, sid in enumerate(ids)}
        pbar = tqdm(total=N, desc="Writing memmap", leave=False)
        written = set()
        for sp in shard_paths:
            sh = torch.load(sp, map_location='cpu')
            clips = sh['clip_patches']
            for i, sid in enumerate(sh['id']):
                if sid in written: continue
                mm[id_to_idx[sid]] = clips[i].numpy()
                written.add(sid)
                pbar.update(1)
        pbar.close(); mm.flush()
        log.info(f"[merge_only] Done. Wrote TSV and memmap: N={N}, shape={(T,Gh,Gw,D)}, dtype={dtype_np}")
        return

    distributed_setup()

    # Helper to shrink features before saving/writing
    def _postprocess_patches(clip_patches: torch.Tensor) -> torch.Tensor:
        # clip_patches: (B, T, Gh, Gw, D)
        B, T, Gh, Gw, D = clip_patches.shape
        # Temporal stride
        if args.temporal_stride and args.temporal_stride > 1:
            clip_patches = clip_patches[:, ::args.temporal_stride]
            B, T, Gh, Gw, D = clip_patches.shape
        # Spatial bilinear downsample on grid
        if args.grid_out and args.grid_out > 0 and (Gh != args.grid_out or Gw != args.grid_out):
            x = clip_patches.permute(0, 1, 4, 2, 3).contiguous()  # (B,T,D,Gh,Gw)
            x = x.view(B * T, D, Gh, Gw)
            x = F.interpolate(x, size=(args.grid_out, args.grid_out), mode='bilinear', align_corners=False)
            x = x.view(B, T, D, args.grid_out, args.grid_out)
            clip_patches = x.permute(0, 1, 3, 4, 2).contiguous()  # (B,T,g,g,D)
        return clip_patches

    # Direct-to-memmap fast path: preallocate memmap on rank 0 and have all ranks write by global index
    if args.direct_memmap:
        # Feature extractor (vision backbone)
        feature_extractor = FeaturesUtils(
            tod_vae_ckpt=None,
            enable_conditions=True,
            synchformer_ckpt=None,
            mode='16k',
            need_vae_encoder=False,
            clip_model=args.clip_model,
        ).eval().cuda()

        dataset, loader = setup_loader(args.video_root, args.tsv, args.batch_size, args.num_workers, args.clip_size)
        N = len(dataset)
        rank = dist.get_rank()

        # Rank 0: decide shape with a quick pilot forward, then create TSV + memmap
        shape_tensor = torch.zeros(4, dtype=torch.int64, device=torch.device('cuda'))
        if rank == 0:
            # Build TSV in dataset order for deterministic indexing
            ids = dataset.videos
            labels = [dataset.labels[v] for v in ids]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': ids, 'label': labels}).to_csv(
                args.output_dir / 'vgg-patches.tsv', sep='\t', index=False)

            # Pilot to get (T, Gh, Gw, D)
            pilot = None
            for j in range(min(8, N)):
                try:
                    x = dataset[j]
                    if x is not None:
                        pilot = x
                        break
                except Exception:
                    continue
            if pilot is None:
                log.error("No decodable sample found for pilot run.")
            else:
                clip = pilot['clip_video'].unsqueeze(0).cuda(non_blocking=True)  # (1, T, C, H, W)
                with torch.inference_mode():
                    y = feature_extractor.encode_video_with_clip_patches(
                        clip, return_grid=True, normalize=False, include_cls=False)
                y = _postprocess_patches(y)
                T, Gh, Gw, D = y.shape[1:]
                shape_tensor[:] = torch.tensor([T, Gh, Gw, D], dtype=torch.int64, device=shape_tensor.device)

                # Create memmap
                out_dir = args.output_dir / 'vgg-patches'
                out_dir.mkdir(parents=True, exist_ok=True)
                dtype_np = np.float16 if args.dtype == 'float16' else np.float32
                mmap_path = out_dir / 'clip_patches.npy'
                np.lib.format.open_memmap(mmap_path, mode='w+', dtype=dtype_np, shape=(N, T, Gh, Gw, D)).flush()

        # Broadcast shape to all ranks
        dist.broadcast(shape_tensor, src=0)
        T, Gh, Gw, D = [int(x) for x in shape_tensor.tolist()]
        if T == 0 or Gh == 0 or Gw == 0 or D == 0:
            log.error("Invalid shape broadcast for memmap; aborting.")
            dist.barrier(); dist.destroy_process_group()
            return

        # All ranks open memmap for writing
        out_dir = args.output_dir / 'vgg-patches'
        mmap_path = out_dir / 'clip_patches.npy'
        dtype_np = np.float16 if args.dtype == 'float16' else np.float32
        mm = np.lib.format.open_memmap(mmap_path, mode='r+', dtype=dtype_np, shape=(N, T, Gh, Gw, D))

        log.info(f"Rank {local_rank}: starting direct writes into memmap with shape (N={N}, T={T}, {Gh}x{Gw}, D={D})")

        successful_batches = 0
        failed_batches = 0
        for curr_iter, data in enumerate(tqdm(loader)):
            if data is None or len(data.get('id', [])) == 0:
                failed_batches += 1
                continue
            try:
                idxs = data['index']  # list[int] global indices into dataset order
                clip_video = data['clip_video'].cuda(non_blocking=True)
                # Extract grid tokens (B, T, Gh, Gw, D)
                clip_patches = feature_extractor.encode_video_with_clip_patches(
                    clip_video, return_grid=True, normalize=False, include_cls=False)
                clip_patches = _postprocess_patches(clip_patches)
                # Write each sample directly to its slot
                cp = clip_patches.detach().cpu().to(dtype=torch.float16 if args.dtype == 'float16' else torch.float32).numpy()
                for bi, gi in enumerate(idxs):
                    mm[gi] = cp[bi]
                successful_batches += 1
            except Exception as e:
                failed_batches += 1
                log.error(f"Rank {local_rank}, iteration {curr_iter}: Error processing batch: {e}")
                continue

        mm.flush()
        log.info(f"Rank {local_rank}: done. {successful_batches} successful, {failed_batches} failed batches")

        dist.barrier()
        dist.destroy_process_group()
        return

    # Note: merge_only path already handled above without distributed init.

    # Feature extractor (only CLIP and tokenizer needed)
    feature_extractor = FeaturesUtils(
        tod_vae_ckpt=None,
        enable_conditions=True,
        synchformer_ckpt=None,  # Not needed for patch extraction
        mode='16k',
        need_vae_encoder=False,
        clip_model=args.clip_model,
    ).eval().cuda()

    dataset, loader = setup_loader(args.video_root, args.tsv, args.batch_size, args.num_workers, args.clip_size)
    log.info(f"Dataset size: {len(dataset)}")

    shard_dir = args.output_dir / f"patch_shards/{local_rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    successful_batches = 0
    failed_batches = 0

    for curr_iter, data in enumerate(tqdm(loader)):
        if data is None or len(data.get('id', [])) == 0:
            failed_batches += 1
            log.warning(f"Rank {local_rank}, iteration {curr_iter}: empty or None batch")
            continue
        
        try:
            ids = data['id']
            clip_video = data['clip_video'].cuda()
            successful_batches += 1
            
            # Extract grid tokens (B, T, Gh, Gw, D)
            clip_patches = feature_extractor.encode_video_with_clip_patches(
                clip_video, return_grid=True, normalize=False, include_cls=False)
            clip_patches = _postprocess_patches(clip_patches)

            # Save a shard to disk (to be merged later)
            shard_path = shard_dir / f"r{local_rank}_{curr_iter}.pth"
            torch.save({
                'id': ids,
                'caption': data['caption'],
                'clip_patches': clip_patches.detach().cpu(),
            }, shard_path)

            # Also write a lightweight metadata TSV to accelerate scanning during merge
            try:
                T, Gh, Gw, D = clip_patches.shape[1:]
                meta_df = pd.DataFrame({
                    'id': ids,
                    'label': data['caption'],
                    'T': [T] * len(ids),
                    'Gh': [Gh] * len(ids),
                    'Gw': [Gw] * len(ids),
                    'D': [D] * len(ids),
                    'dtype': [str(clip_patches.dtype).replace('torch.', '')] * len(ids),
                })
                meta_df.to_csv(shard_path.with_suffix('.meta.tsv'), sep='\t', index=False)
            except Exception:
                pass
            
        except Exception as e:
            failed_batches += 1
            log.error(f"Rank {local_rank}, iteration {curr_iter}: Error processing batch: {e}")
            continue

    log.info(f"Rank {local_rank}: {successful_batches} successful, {failed_batches} failed batches")

    dist.barrier()

    # Merge per-rank shards to a single on-disk array on rank 0 (streaming to avoid OOM)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        log.info("Combining patch shards into a memmap directory (streaming)...")
        base = args.output_dir / "patch_shards"
        shard_paths = []
        meta_paths = []
        for rank_name in sorted(os.listdir(base)):
            rank_dir = base / rank_name
            for shard_name in sorted(os.listdir(rank_dir)):
                sp = rank_dir / shard_name
                if sp.suffix == '.pth':
                    shard_paths.append(str(sp))
                elif shard_name.endswith('.meta.tsv'):
                    meta_paths.append(str(sp))

        # First pass: count unique ids and capture shape/dtype
        used_id = set()
        ids = []
        labels = []
        sample_shape = None
        sample_dtype = None
        if len(meta_paths) > 0:
            for mp in tqdm(meta_paths, desc="Scanning shard metadata", leave=False):
                mdf = pd.read_csv(mp, sep='\t')
                if sample_shape is None and {'T','Gh','Gw','D','dtype'}.issubset(mdf.columns):
                    sample_shape = (int(mdf['T'][0]), int(mdf['Gh'][0]), int(mdf['Gw'][0]), int(mdf['D'][0]))
                    sample_dtype = str(mdf['dtype'][0])
                for _, row in mdf.iterrows():
                    sid = row['id']
                    if sid in used_id:
                        continue
                    used_id.add(sid)
                    ids.append(sid)
                    labels.append(row['label'])
        else:
            for sp in tqdm(shard_paths, desc="Scanning shards", leave=False):
                sh = torch.load(sp, map_location='cpu')  # avoid weights_only to keep lists
                clips = sh['clip_patches']  # (B, T, Gh, Gw, D)
                if sample_shape is None:
                    sample_shape = tuple(clips.shape[1:])
                    sample_dtype = str(clips.dtype).replace('torch.', '')  # e.g., 'float16'
                for i, sid in enumerate(sh['id']):
                    if sid in used_id:
                        continue
                    used_id.add(sid)
                    ids.append(sid)
                    labels.append(sh['caption'][i])
        N = len(ids)
        if N == 0:
            log.warning("No samples found to merge. Skipping memmap creation.")
        else:
            # Write TSV
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': ids, 'label': labels}).to_csv(
                args.output_dir / 'vgg-patches.tsv', sep='\t', index=False)

            # Create numpy memmap and stream-write sample by sample
            T, Gh, Gw, D = sample_shape
            dtype_np = np.float16 if 'float16' in sample_dtype else np.float32
            out_dir = args.output_dir / 'vgg-patches'
            out_dir.mkdir(parents=True, exist_ok=True)
            mmap_path = out_dir / 'clip_patches.npy'
            mm = np.lib.format.open_memmap(mmap_path, mode='w+', dtype=dtype_np, shape=(N, T, Gh, Gw, D))

            # Second pass: write
            id_to_idx = {sid: i for i, sid in enumerate(ids)}
            written = set()
            pbar = tqdm(total=N, desc="Writing memmap", leave=False)
            for sp in shard_paths:
                sh = torch.load(sp, map_location='cpu')
                clips = sh['clip_patches']
                for i, sid in enumerate(sh['id']):
                    if sid in written:
                        continue
                    idx = id_to_idx[sid]
                    mm[idx] = clips[i].numpy()
                    written.add(sid)
                    pbar.update(1)
            pbar.close()
            mm.flush()
            log.info(f"Wrote TSV and memmap: N={N}, shape={(T, Gh, Gw, D)}, dtype={dtype_np}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
