import os, random
import numpy as np
import json
import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / 'pavas'
CACHE_ROOT = Path(tempfile.gettempdir()) / 'pavas-cache'
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_ROOT / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(CACHE_ROOT / 'numba'))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import torch.distributed as dist

from core.utils.logger import TensorboardLogger
from core.runner import Runner
from core.data.data_setup import setup_test_datasets
from core.model.sequence_config import SequenceConfig, CONFIG_16K, CONFIG_44K
from hydra.core.hydra_config import HydraConfig


def distributed_setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def distributed_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Some operations may become slower when deterministic mode is enabled.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

@hydra.main(version_base="1.3.2", config_path="configs", config_name="eval/eval.yaml")
def main(cfg: DictConfig):
    local_rank = distributed_setup()

    # Use a rank-aware seed and fall back to 1234 when cfg.seed is missing.
    base_seed = int(getattr(cfg, "seed", 1234))
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    set_seed(base_seed + rank)

    run_dir = Path(os.getcwd())

    # Select the base configuration from the requested model variant.
    if str(cfg.model).endswith('16k'):
        base = CONFIG_16K
    elif str(cfg.model).endswith('44k'):
        base = CONFIG_44K
    else:
        raise ValueError(f"Unknown model variant: {cfg.model}")

    # Build the sequence configuration from the requested duration.
    duration_s = float(getattr(cfg, 'duration_s', 8.0))
    seq_cfg = SequenceConfig(
        duration=duration_s,
        sampling_rate=base.sampling_rate,
        spectrogram_frame_rate=base.spectrogram_frame_rate,
        clip_frame_rate=base.clip_frame_rate,
        sync_frame_rate=base.sync_frame_rate,
        sync_num_frames_per_segment=base.sync_num_frames_per_segment,
        sync_step_size=base.sync_step_size,
        latent_downsample_rate=base.latent_downsample_rate
    )

    # Populate data_dim with the sequence lengths and feature dimensions.
    # text_len defaults to 77, and the feature dims default to clip/text=1024 and sync=768.
    text_len = 77
    try:
        if 'data_dim' in cfg and 'text_seq_len' in cfg.data_dim:
            text_len = int(cfg.data_dim.text_seq_len)
        elif 'text_seq_len' in cfg:
            text_len = int(cfg.text_seq_len)
    except Exception:
        pass
    # Use cfg-provided dims when present, otherwise fall back to the release defaults.
    def get_dd(key, default):
        try:
            return int(cfg.data_dim.get(key, default))  # type: ignore[attr-defined]
        except Exception:
            return default
    clip_dim = get_dd('clip_dim', 1024)
    sync_dim = get_dd('sync_dim', 768)
    text_dim = get_dd('text_dim', 1024)

    # Adjust the batch size for multi-GPU runs.
    try:
        world = dist.get_world_size() if dist.is_initialized() else 1
    except Exception:
        world = 1

    with open_dict(cfg):
        cfg.data_dim = {
            'latent_seq_len': seq_cfg.latent_seq_len,
            'clip_seq_len': seq_cfg.clip_seq_len,
            'sync_seq_len': seq_cfg.sync_seq_len,
            'num_audio_frames': seq_cfg.num_audio_frames,
            'text_seq_len': text_len,
            'clip_dim': clip_dim,
            'sync_dim': sync_dim,
            'text_dim': text_dim,
        }
        if 'batch_size' in cfg and world > 1:
            cfg.batch_size = max(1, int(cfg.batch_size) // world)

        cfg.enable_video_preview = False  # Disable video preview synthesis during evaluation.

    # Logger; W&B settings are read from cfg.wandb when available.
    wandb_cfg = None
    try:
        wandb_cfg = cfg.wandb
    except Exception:
        pass
    logger = TensorboardLogger(
        cfg.exp_id,
        run_dir,
        logging.getLogger(),
        is_rank0=(local_rank == 0),
        enable_email=False,
        wandb_cfg=wandb_cfg,
    )

    runner = Runner(cfg, log=logger, run_path=run_dir, for_training=False)

    weight_path = getattr(getattr(cfg, "test", {}), "weight_path", None)
    if weight_path is None:
        if local_rank == 0:
            logger.info("Error: provide +test.weight_path=/path/to/weights.pth")
        distributed_cleanup()
        return

    runner.load_weights(weight_path)
    runner.enter_val()

    # Output root can be overridden via +output_root=..., otherwise Hydra run.dir is used.
    try:
        hydra_run_dir = Path(HydraConfig.get().run.dir)
    except Exception:
        hydra_run_dir = run_dir
    output_root = Path(getattr(cfg, "output_root", hydra_run_dir))
    out_dir = output_root / str(cfg.exp_id) / "test-sampled"
    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset, sampler, loader = setup_test_datasets(cfg)

    gt_cache = getattr(cfg.data.ExtractedVGG_test, "gt_cache", None)
    data_cfg = OmegaConf.create({
        "tag": "test",
        "gt_cache": gt_cache,
        "compute_metrics": getattr(cfg.data.ExtractedVGG_test, "compute_metrics", True),
        # Pass an absolute path so the logger can use it directly.
        "output_subdir": str(out_dir),
    })

    metrics = runner.evaluate_dataset(loader, data_cfg, it=0, force_preview=False)

    if local_rank == 0 and metrics is not None:
        print("Test metrics:", metrics)
        try:
            (run_dir / "test_metrics.json").write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))
        except Exception:
            pass

    distributed_cleanup()

if __name__ == "__main__":
    main()
