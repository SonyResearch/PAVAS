import logging
import math
import os
import random
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

os.environ.setdefault('LOCAL_RANK', '0')

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / 'pavas'
CACHE_ROOT = Path(tempfile.gettempdir()) / 'pavas-cache'
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_ROOT / 'matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(CACHE_ROOT / 'numba'))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from core.data.data_setup import setup_training_datasets, setup_val_datasets
from core.model.sequence_config import CONFIG_16K, CONFIG_44K
from core.runner import Runner
from core.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from core.utils.logger import TensorboardLogger
from core.utils.synthesize_ema import synthesize_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


@record
@hydra.main(version_base='1.3.2', config_path='configs', config_name='train/finetune.yaml')
def train(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # The runner still uses DDP in single-process debug mode.
    debug_single = bool(getattr(cfg, 'debug_single_process', False))
    debug_store = None
    if debug_single:
        if world_size != 1:
            raise ValueError('debug_single_process requires WORLD_SIZE=1')
        debug_store = tempfile.TemporaryDirectory(prefix='pavas-debug-dist-')
        distributed.init_process_group(
            backend='nccl',
            init_method=f'file://{debug_store.name}/rendezvous',
            rank=0,
            world_size=1,
            timeout=timedelta(hours=2),
        )
        logging.getLogger().info('Debug single-process mode: initialized local DDP group')
    else:
        distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # compose early such that it does not rely on future hard disk reading
    eval_cfg = compose('eval_config', overrides=[f'exp_id={cfg.exp_id}'])

    # patch data dim
    if cfg.model.endswith('16k'):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith('44k'):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Unknown model: {cfg.model}')
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # wrap python logger with a tensorboard logger
    # Extract wandb configuration (may be missing in old configs)
    wandb_cfg = None
    try:
        wandb_cfg = dict(cfg.wandb)
    except Exception:
        wandb_cfg = None

    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=cfg.enable_email and not cfg.debug,
                            wandb_cfg=wandb_cfg)

    # Propagate object pruning settings to environment (mutually exclusive: per_prompt_top_k vs global_top_k)
    try:
        oc_obj_prune = getattr(getattr(cfg, 'object_cond', {}), 'obj_prune', {}) or {}
        gk = oc_obj_prune.get('global_top_k', None)
        pp = oc_obj_prune.get('per_prompt_top_k', None)
        import os
        # Clear all related env first for determinism
        for k in ['MMAUDIO_PER_PROMPT_TOPK', 'MMAUDIO_GLOBAL_OBJ_TOPK', 'MMAUDIO_DIVERSE_RR', 'MMAUDIO_PER_CLASS_TOPK']:
            if k in os.environ:
                del os.environ[k]
        if gk and pp:
            raise ValueError(f"Both global_top_k({gk}) and per_prompt_top_k({pp}) are >0. Choose one.")
        if pp:
            os.environ['MMAUDIO_PER_PROMPT_TOPK'] = str(int(pp))
            info_if_rank_zero(log, f'[object_cond] Per-prompt top-k enabled: K_prompt={pp}')
        elif gk:
            os.environ['MMAUDIO_GLOBAL_OBJ_TOPK'] = str(int(gk))
            info_if_rank_zero(log, f'[object_cond] Global object top-k pruning enabled: K={gk}')
        else:
            info_if_rank_zero(log, '[object_cond] No object pruning (K unset)')
    except Exception as _e:
        info_if_rank_zero(log, f'[object_cond] Failed to set pruning env vars: {_e}')

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Training configuration: {cfg}')
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')

    # determine time to change max skip
    total_iterations = cfg['num_iterations']

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset)}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    val_dataset, val_loader, eval_loader = setup_val_datasets(cfg)
    info_if_rank_zero(log, f'Number of val samples: {len(val_dataset)}')
    val_cfg = cfg.data.ExtractedVGG_val

    # compute and set mean and std
    latent_mean, latent_std = dataset.compute_latent_stats()

    # construct the trainer
    trainer = Runner(cfg,
                     log=log,
                     run_path=run_dir,
                     for_training=True,
                     latent_mean=latent_mean,
                     latent_std=latent_std).enter_train()
    # Optionally watch model parameters/gradients (wandb)
    try:
        if getattr(cfg, 'wandb', None) and getattr(cfg.wandb, 'enable', False) and getattr(cfg.wandb, 'watch_model', False):
            import wandb  # local import to avoid hard dependency when disabled
            if log.wandb_enabled:
                wandb.watch(trainer.network.module, log='gradients', log_freq=cfg.log_text_interval)
    except Exception as e:
        log.warning(f'wandb watch failed: {e}')
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # load previous checkpoint if needed
    if cfg['checkpoint'] is not None:
        curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
        cfg['checkpoint'] = None
        info_if_rank_zero(log, 'Model checkpoint loaded!')
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, 'Latest checkpoint loaded!')
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg['weights'] is None:
                raise ValueError('Provide weights=/path/to/mmaudio_checkpoint.pth to initialize PAVAS training.')
            if not Path(cfg['weights']).is_file():
                raise FileNotFoundError(f"Backbone initialization weights not found: {cfg['weights']}")
            info_if_rank_zero(log, 'Loading weights from the disk')
            trainer.load_weights(cfg['weights'])
            cfg['weights'] = None

    # determine max epoch
    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log, f'We will approximately use {total_epoch} epochs.')

    # Optional one-time evaluation before the first training update.
    try:
        if bool(getattr(cfg, 'eval_at_start', False)):
            info_if_rank_zero(log, '[startup] Running evaluation before any training updates')
            it0 = curr_iter
            train_rng_snapshot = trainer.rng.graphsafe_get_state()
            trainer.rng.graphsafe_set_state(eval_rng_clone)

            processed = 0
            max_batches = getattr(cfg, 'eval_max_batches', None)
            for b_idx, data in enumerate(eval_loader):
                audio_path = trainer.inference_pass(
                    data,
                    it=it0,
                    data_cfg=val_cfg,
                    save_eval=True,
                    is_first_eval_batch=(b_idx == 0),
                    force_preview=True
                )
                processed += 1
                if (max_batches is not None) and (processed >= int(max_batches)):
                    break

            if not debug_single:
                distributed.barrier()
            trainer.rng.graphsafe_set_state(train_rng_snapshot)
            trainer.eval(audio_path, it0, val_cfg)
    except Exception as _e:
        info_if_rank_zero(log, f'[startup] initial eval failed: {_e}')

    # training loop
    try:
        # Need this to select random bases in different workers
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            # Crucial for randomness!
            sampler.set_epoch(current_epoch)
            current_epoch += 1
            log.debug(f'Current epoch: {current_epoch}')

            trainer.enter_train()
            trainer.log.data_timer.start()
            for data in loader:
                trainer.train_pass(data, curr_iter)

                if (curr_iter + 1) % cfg.val_interval == 0:
                    # swap into a eval rng state, i.e., use the same seed for every validation pass
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f'Iteration {curr_iter}: validating')
                    for data in val_loader:
                        trainer.validation_pass(data, curr_iter)
                    if not debug_single:
                        distributed.barrier()
                    trainer.val_integrator.finalize('val', curr_iter, ignore_timer=True)
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)

                if (curr_iter + 1) % cfg.eval_interval == 0:
                    save_eval = (curr_iter + 1) % cfg.save_eval_interval == 0
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f'Iteration {curr_iter}: validating')
                    processed = 0
                    max_batches = getattr(cfg, 'eval_max_batches', None)
                    for b_idx, data in enumerate(eval_loader):
                        audio_path = trainer.inference_pass(data,
                                                            curr_iter,
                                                            val_cfg,
                                                            save_eval=save_eval,
                                                            is_first_eval_batch=(b_idx == 0))
                        processed += 1
                        if (max_batches is not None) and (processed >= int(max_batches)):
                            break
                    if not debug_single:
                        distributed.barrier()
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)
                    trainer.eval(audio_path, curr_iter, val_cfg)

                curr_iter += 1

                if curr_iter >= total_iterations:
                    break
    except Exception as e:
        log.error(f'Error occurred at iteration {curr_iter}!')
        log.critical(e.message if hasattr(e, 'message') else str(e))
        raise
    finally:
        if not cfg.debug:
            trainer.save_checkpoint(curr_iter)
            trainer.save_weights(curr_iter)

    # In debug single-process mode, skip EMA and evaluation
    if bool(getattr(cfg, 'debug_single_process', False)):
        log.info('Debug single-process mode: skipping EMA synthesis and evaluation.')
        # clean-up logger
        log.complete()
        distributed.destroy_process_group()
        debug_store.cleanup()
        return

    # Inference pass
    del trainer
    torch.cuda.empty_cache()

    # Synthesize EMA
    if local_rank == 0:
        log.info(f'Synthesizing EMA with sigma={cfg.ema.default_output_sigma}')
        ema_sigma = cfg.ema.default_output_sigma
        state_dict = synthesize_ema(cfg, ema_sigma, step=None)
        if state_dict is None:
            log.warning('Skipping final EMA synthesis because no EMA checkpoints were written yet.')
        else:
            save_dir = Path(run_dir) / f'{cfg.exp_id}_ema_final.pth'
            torch.save(state_dict, save_dir)
            log.info(f'Synthesized EMA saved to {save_dir}!')
    if not debug_single:
        distributed.barrier()

    log.info(f'Evaluation: {eval_cfg}')
    from core.sample import sample  # lazy import to avoid env issues in debug mode
    try:
        sample(eval_cfg)
    except FileNotFoundError as exc:
        log.warning(f'Skipping post-training sample because evaluation assets are unavailable: {exc}')

    # clean-up
    log.complete()
    if not debug_single:
        distributed.barrier()
        distributed.destroy_process_group()


if __name__ == '__main__':
    train()
