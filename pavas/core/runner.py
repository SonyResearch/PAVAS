"""
trainer.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""
import os
import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from core.model.flow_matching import FlowMatching
from core.model.networks import get_my_mmaudio
from core.model.sequence_config import CONFIG_16K, CONFIG_44K
from core.model.utils.features_utils import FeaturesUtils
from core.model.utils.parameter_groups import get_parameter_groups
from core.model.utils.sample_utils import log_normal_sample
from core.utils.dist_utils import (info_if_rank_zero, local_rank, string_if_rank_zero)
from core.utils.log_integrator import Integrator
from core.utils.logger import TensorboardLogger
from core.utils.time_estimator import PartialTimeEstimator, TimeEstimator
from core.utils.video_joiner import VideoJoiner
from core.utils.io_profile import ENABLE_IO_PROFILE, io_profile_snapshot, format_snapshot, io_timer, io_add
from tqdm import tqdm
import time

try:
    import wandb  # optional
except Exception:
    wandb = None

class Runner:

    def __init__(self,
                 cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: Union[str, Path],
                 for_training: bool = True,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None):
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.enable_grad_scaler = cfg.enable_grad_scaler
        self.for_training = for_training
        self.cfg = cfg
        self.log = log

        if cfg.model.endswith('16k'):
            self.seq_cfg = CONFIG_16K
            mode = '16k'
        elif cfg.model.endswith('44k'):
            self.seq_cfg = CONFIG_44K
            mode = '44k'
        else:
            raise ValueError(f'Unknown model: {cfg.model}')

        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration

        # setting up the model
        empty_string_feat = torch.load('./ext_weights/empty_string.pth', weights_only=True)[0]
        self.network = DDP(
            get_my_mmaudio(
                cfg.model,
                latent_mean=latent_mean,
                latent_std=latent_std,
                empty_string_feat=empty_string_feat,
                object_cond_cfg=getattr(cfg, 'object_cond', None),
                architecture_debug=getattr(cfg, 'architecture_debug', False),
                use_delta_mod=getattr(cfg, 'use_delta_mod', False),  # Toggle passed from config.
            ).cuda(),
            device_ids=[local_rank],
            broadcast_buffers=False,
            # OC branches (mass/velocity/patch) may be skipped on some steps; allow unused params
            find_unused_parameters=True,
        )
        # Disable compile when architecture debugging is enabled; diagnostic prints disrupt Inductor/AOT.
        should_compile = bool(getattr(cfg, 'compile', False)) and not bool(
            getattr(cfg, 'architecture_debug', False))
        if should_compile:
            # NOTE: though train_fn and val_fn are very similar
            # (early on they are implemented as a single function)
            # keeping them separate and compiling them separately are CRUCIAL for high performance
            self.train_fn = torch.compile(self.train_fn)
            self.val_fn = torch.compile(self.val_fn)

        self.fm = FlowMatching(cfg.sampling.min_sigma,
                               inference_mode=cfg.sampling.method,
                               num_steps=cfg.sampling.num_steps)

        # ema profile
        if for_training and cfg.ema.enable and local_rank == 0:
            self.ema = PostHocEMA(self.network.module,
                                  sigma_rels=cfg.ema.sigma_rels,
                                  update_every=cfg.ema.update_every,
                                  checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                                  checkpoint_folder=cfg.ema.checkpoint_folder,
                                  step_size_correction=True).cuda()
            self.ema_start = cfg.ema.start
        else:
            self.ema = None

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(cfg['seed'] + local_rank)

        if ENABLE_IO_PROFILE and local_rank == 0:
            self.log.info('[io-prof] I/O profiling ENABLED (env MMAUDIO_PROFILE_IO). Snapshots will emit every log_text_interval.')

        # setting up feature extractors and VAEs
        if mode == '16k':
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_16k_ckpt'],
                bigvgan_vocoder_ckpt=cfg['bigvgan_vocoder_ckpt'],
                synchformer_ckpt=cfg['synchformer_ckpt'],
                enable_conditions=True,
                clip_model='vit_h_14_384',
                mode=mode,
                need_vae_encoder=False,
            )
        elif mode == '44k':
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_44k_ckpt'],
                synchformer_ckpt=cfg['synchformer_ckpt'],
                enable_conditions=True,
                clip_model='vit_h_14_384',
                mode=mode,
                need_vae_encoder=False,
            )
        self.features = self.features.cuda().eval()

        # Optionally compile features (VAE/vocoder). Disabled by default to avoid Inductor issues in logging.
        compile_features = bool(getattr(cfg, 'compile_features', False))
        if should_compile and compile_features:
            self.features.compile()

        # hyperparameters
        self.log_normal_sampling_mean = cfg.sampling.mean
        self.log_normal_sampling_scale = cfg.sampling.scale
        self.null_condition_probability = cfg.null_condition_probability
        # Separate classifier-free dropout probability for object conditions (mass/velocity/patch)
        self.null_object_condition_probability = float(getattr(cfg, 'null_object_condition_probability', 0.0))
        self.cfg_strength = cfg.cfg_strength

        # setting up logging
        # self.log = log
        self.run_path = Path(run_path)
        vgg_cfg = cfg.data.VGGSound
        if for_training:
            self.val_video_joiner = VideoJoiner(vgg_cfg.root, self.run_path / 'val-sampled-videos',
                                                self.sample_rate, self.duration_sec)
        else:
            self.test_video_joiner = VideoJoiner(vgg_cfg.root,
                                                 self.run_path / 'test-sampled-videos',
                                                 self.sample_rate, self.duration_sec)
        string_if_rank_zero(self.log, 'model_size',
                            f'{sum([param.nelement() for param in self.network.parameters()])}')
        string_if_rank_zero(
            self.log, 'number_of_parameters_that_require_gradient: ',
            str(
                sum([
                    param.nelement()
                    for param in filter(lambda p: p.requires_grad, self.network.parameters())
                ])))
        info_if_rank_zero(self.log, 'torch version: ' + torch.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        # Diagnostics & visualization config (compile-safe, rank0 only)
        diag_cfg = getattr(cfg, 'diagnostics', None)
        self.diag_enabled = bool(getattr(diag_cfg, 'enabled', True)) if diag_cfg is not None else True
        # Default every_n aligns with extra logging cadence if available
        default_every_n = int(getattr(cfg, 'log_extra_interval', 50)) if hasattr(cfg, 'log_extra_interval') else 50
        self.diag_every_n = int(getattr(diag_cfg, 'every_n', default_every_n)) if diag_cfg is not None else default_every_n
        # Coverage thresholds for presence agreement
        self.diag_cov_thresholds = list(getattr(diag_cfg, 'coverage_thresholds', [0.5])) if diag_cfg is not None else [0.5]
        # Lightweight visualization flags
        self.vis_enabled = bool(getattr(diag_cfg, 'visualize', False)) if diag_cfg is not None else False
        self.vis_every_n = int(getattr(diag_cfg, 'visualize_every_n', self.diag_every_n)) if diag_cfg is not None else self.diag_every_n

        # setting up optimizer and loss
        if for_training:
            self.enter_train()
            parameter_groups = get_parameter_groups(self.network, cfg, print_log=(local_rank == 0))
            self.optimizer = optim.AdamW(parameter_groups,
                                         lr=cfg['learning_rate'],
                                         weight_decay=cfg['weight_decay'],
                                         betas=[0.9, 0.95],
                                         eps=1e-6 if self.use_amp else 1e-8,
                                         fused=True)
            if self.enable_grad_scaler:
                self.scaler = torch.amp.GradScaler(init_scale=2048)
            self.clip_grad_norm = cfg['clip_grad_norm']

            # linearly warmup learning rate
            linear_warmup_steps = cfg['linear_warmup_steps']

            def warmup(currrent_step: int):
                return (currrent_step + 1) / (linear_warmup_steps + 1)

            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)

            # setting up learning rate scheduler
            if cfg['lr_schedule'] == 'constant':
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
            elif cfg['lr_schedule'] == 'poly':
                total_num_iter = cfg['iterations']
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                             lr_lambda=lambda x:
                                                             (1 - (x / total_num_iter))**0.9)
            elif cfg['lr_schedule'] == 'step':
                next_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                cfg['lr_schedule_steps'],
                                                                cfg['lr_schedule_gamma'])
            else:
                raise NotImplementedError

            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                             [warmup_scheduler, next_scheduler],
                                                             [linear_warmup_steps])

            # Logging info
            self.log_text_interval = cfg['log_text_interval']
            self.log_extra_interval = cfg['log_extra_interval']
            self.save_weights_interval = cfg['save_weights_interval']
            self.save_checkpoint_interval = cfg['save_checkpoint_interval']
            self.save_copy_iterations = cfg['save_copy_iterations']
            self.num_iterations = cfg['num_iterations']
            if cfg['debug']:
                self.log_text_interval = self.log_extra_interval = 1

            # update() is called when we log metrics, within the logger
            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
            # update() is called every iteration, in this script
            self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)
        else:
            self.enter_val()

    def _extract_object_cond(self, data) -> dict:
        """
        Extract object-condition tensors from the batch and move them to CUDA.
        Missing keys are skipped for backward compatibility.
        """
        cond = {}
        if 'velocity_f' in data:
            cond['velocity_f'] = data['velocity_f'].cuda(non_blocking=True)
        if 'mass_f' in data:
            cond['mass_f'] = data['mass_f'].cuda(non_blocking=True)
        if 'presence_mask' in data:
            cond['presence_mask'] = data['presence_mask'].cuda(non_blocking=True)
        if 'clip_patches' in data:
            cond['clip_patches'] = data['clip_patches'].cuda(non_blocking=True)
        if 'obj_patch_attn' in data:
            cond['obj_patch_attn'] = data['obj_patch_attn'].cuda(non_blocking=True)
        if 'obj_empty_mask' in data:
            cond['obj_empty_mask'] = data['obj_empty_mask'].cuda(non_blocking=True)
        if 'velocity_missing' in data:
            cond['velocity_missing'] = data['velocity_missing'].cuda(non_blocking=True)
        # Optional bookkeeping tensors (kept on CPU unless needed on GPU later)
        if 'obj_label_ids' in data:
            cond['obj_label_ids'] = data['obj_label_ids']  # int64 (Omax,) per sample broadcast after padding (not yet padded here)
        if 'obj_prompt_ids' in data:
            cond['obj_prompt_ids'] = data['obj_prompt_ids']
        return cond

    def train_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
        *,
        cond: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        x1 = a_mean + a_std * a_randn
        bs = x1.shape[0]  # batch_size * seq_len * num_channels

        # normalize the latents
        x1 = self.network.module.normalize(x1)

        t = log_normal_sample(x1,
                              generator=self.rng,
                              m=self.log_normal_sampling_mean,
                              s=self.log_normal_sampling_scale)
        x0, x1, xt, (clip_f, sync_f, text_f) = self.fm.get_x0_xt_c(x1,
                                                                   t,
                                                                   Cs=[clip_f, sync_f, text_f],
                                                                   generator=self.rng)

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_video = (samples < self.null_condition_probability)
        if null_video.any():
            vmask = null_video.view(-1, 1, 1)
            empty_clip = (self.network.module.empty_clip_feat
                          .unsqueeze(0)
                          .expand(bs, clip_f.size(1), -1))
            empty_sync = (self.network.module.empty_sync_feat
                          .unsqueeze(0)
                          .expand(bs, sync_f.size(1), -1))
            clip_f = torch.where(vmask, empty_clip, clip_f)
            sync_f = torch.where(vmask, empty_sync, sync_f)

        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        if null_text.any():
            tmask = null_text.view(-1, 1, 1)
            empty_text = (self.network.module.empty_string_feat
                          .unsqueeze(0)
                          .expand(bs, text_f.size(1), -1))
            text_f = torch.where(tmask, empty_text, text_f)

        # Forward object conditions only when they are available.
        cond = cond or {}
        if self.null_object_condition_probability > 0.0 and any(
                k in cond for k in ('velocity_f', 'mass_f', 'obj_patch_attn', 'presence_mask')):
            with torch.no_grad():
                # `live_video` means the sample kept its video condition after CFG dropout.
                live_video = ~null_video
                video_exist_mask = cond.get('video_exist')  # Optional `(B,)` boolean mask.
                if video_exist_mask is not None:
                    live_video = live_video & video_exist_mask
                samples_obj = torch.rand(bs, device=x1.device, generator=self.rng)
                obj_drop_live = (samples_obj < self.null_object_condition_probability) & live_video
                # Disable object conditions for non-live samples or dropped live samples.
                object_null_mask = (~live_video) | obj_drop_live
            cond['object_null_mask'] = object_null_mask
        try:
            pred_v = self.network(xt, clip_f, sync_f, text_f, t, **cond)
        except TypeError:
            pred_v = self.network(xt, clip_f, sync_f, text_f, t)

        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()
        return x1, loss, mean_loss, t

    def val_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        x1: torch.Tensor,
        *,
        cond: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x1.shape[0]  # batch_size * seq_len * num_channels
        # normalize the latents
        x1 = self.network.module.normalize(x1)
        t = log_normal_sample(x1,
                              generator=self.rng,
                              m=self.log_normal_sampling_mean,
                              s=self.log_normal_sampling_scale)
        x0, x1, xt, (clip_f, sync_f, text_f) = self.fm.get_x0_xt_c(x1,
                                                                   t,
                                                                   Cs=[clip_f, sync_f, text_f],
                                                                   generator=self.rng)

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        # null mask is for when a video is provided but we decided to ignore it
        null_video = (samples < self.null_condition_probability)
        # complete mask is for when a video is not provided or we decided to ignore it
        if null_video.any():
            vmask = null_video.view(-1, 1, 1)
            empty_clip = (self.network.module.empty_clip_feat
                          .unsqueeze(0)
                          .expand(bs, clip_f.size(1), -1))
            empty_sync = (self.network.module.empty_sync_feat
                          .unsqueeze(0)
                          .expand(bs, sync_f.size(1), -1))
            clip_f = torch.where(vmask, empty_clip, clip_f)
            sync_f = torch.where(vmask, empty_sync, sync_f)

        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        if null_text.any():
            tmask = null_text.view(-1, 1, 1)
            empty_text = (self.network.module.empty_string_feat
                          .unsqueeze(0)
                          .expand(bs, text_f.size(1), -1))
            text_f = torch.where(tmask, empty_text, text_f)

        cond = cond or {}
        if self.null_object_condition_probability > 0.0 and any(
                k in cond for k in ('velocity_f', 'mass_f', 'obj_patch_attn', 'presence_mask')):
            with torch.no_grad():
                live_video = ~null_video
                video_exist_mask = cond.get('video_exist')
                if video_exist_mask is not None:
                    live_video = live_video & video_exist_mask
                samples_obj = torch.rand(bs, device=x1.device, generator=self.rng)
                obj_drop_live = (samples_obj < self.null_object_condition_probability) & live_video
                object_null_mask = (~live_video) | obj_drop_live
            cond['object_null_mask'] = object_null_mask
        try:
            pred_v = self.network(xt, clip_f, sync_f, text_f, t, **cond)
        except TypeError:
            pred_v = self.network(xt, clip_f, sync_f, text_f, t)

        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()
        return loss, mean_loss, t

    def train_pass(self, data, it: int = 0):

        if not self.for_training:
            raise ValueError('train_pass() should not be called when not training.')

        self.enter_train()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            # Host->Device transfer timing
            with io_timer('h2d_batch'):
                clip_f = data['clip_features'].cuda(non_blocking=True)
                sync_f = data['sync_features'].cuda(non_blocking=True)
                text_f = data['text_features'].cuda(non_blocking=True)
                video_exist = data['video_exist'].cuda(non_blocking=True)
                text_exist = data['text_exist'].cuda(non_blocking=True)
                a_mean = data['a_mean'].cuda(non_blocking=True)
                a_std = data['a_std'].cuda(non_blocking=True)

            # these masks are for non-existent data; use torch.where to avoid in-place masked writes
            if (~video_exist).any():
                vmask = (~video_exist).view(-1, 1, 1)
                empty_clip = (self.network.module.empty_clip_feat
                              .unsqueeze(0)
                              .expand(clip_f.size(0), clip_f.size(1), -1))
                empty_sync = (self.network.module.empty_sync_feat
                              .unsqueeze(0)
                              .expand(sync_f.size(0), sync_f.size(1), -1))
                clip_f = torch.where(vmask, empty_clip, clip_f)
                sync_f = torch.where(vmask, empty_sync, sync_f)
            if (~text_exist).any():
                tmask = (~text_exist).view(-1, 1, 1)
                empty_text = (self.network.module.empty_string_feat
                              .unsqueeze(0)
                              .expand(text_f.size(0), text_f.size(1), -1))
                text_f = torch.where(tmask, empty_text, text_f)

            # Extract object conditions.
            cond = self._extract_object_cond(data)
            # Keep video_exist in cond so it can be used for object CFG masking.
            try:
                cond['video_exist'] = video_exist  # (B,) bool
            except Exception:
                pass

            self.log.data_timer.end()
            if it % self.log_extra_interval == 0:
                unmasked_clip_f = clip_f.clone()
                unmasked_sync_f = sync_f.clone()
                unmasked_text_f = text_f.clone()
                unmasked_cond = {k: v.clone() for k, v in cond.items()}
            # Forward / loss timing (excludes optimizer update)
            with io_timer('forward_train'):
                x1, loss, mean_loss, t = self.train_fn(clip_f, sync_f, text_f, a_mean, a_std, cond=cond)

            # Integrate aggregated per-batch profiling (originated in worker processes)
            if ENABLE_IO_PROFILE and '__io_prof' in data:
                prof_batch = data['__io_prof']
                for k,v in prof_batch.items():
                    io_add(k, v)

            self.train_integrator.add_dict({'loss': mean_loss})

        if it % self.log_text_interval == 0 and it != 0:
            self.train_integrator.add_scalar('lr', self.scheduler.get_last_lr()[0])
            self.train_integrator.add_binned_tensor('binned_loss', loss, t)
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()
            # I/O profiling snapshot (optional, only when enabled via env MMAUDIO_PROFILE_IO)
            if ENABLE_IO_PROFILE:
                try:
                    snap = io_profile_snapshot()
                    # Always log (even if empty) so user knows profiling path executed
                    if snap:
                        self.log.info('[io-prof] ' + format_snapshot(snap))
                    else:
                        self.log.info('[io-prof] (no timed sections recorded in this interval)')
                except Exception as e:
                    self.log.warning(f'IO profile snapshot failed: {e}')

        # Compile-safe diagnostics and optional visualization (rank0 only)
        if (local_rank == 0
                and self.diag_enabled
                and it % self.diag_every_n == 0
                and it != 0):
            with torch.no_grad():
                try:
                    self._log_object_condition_diagnostics(cond, it)
                    if self.vis_enabled and (it % self.vis_every_n == 0):
                        self._log_object_condition_visualization(cond, it)
                except Exception as e:
                    self.log.warning(f'Diagnostics/visualization failed at it={it}: {e}')

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar('grad_norm', grad_norm)

        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp,
                                dtype=torch.bfloat16), torch.inference_mode():
            try:
                if it % self.log_extra_interval == 0:
                    # save GT audio
                    # unnormalize the latents
                    x1 = self.network.module.unnormalize(x1[0:1])
                    mel = self.features.decode(x1)
                    audio = self.features.vocode(mel).cpu()[0]  # 1 * num_samples
                    self.log.log_spectrogram('train', f'spec-gt-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-gt-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)

                    # save audio from sampling
                    x0 = torch.empty_like(x1[0:1]).normal_(generator=self.rng)
                    clip_f = unmasked_clip_f[0:1]
                    sync_f = unmasked_sync_f[0:1]
                    text_f = unmasked_text_f[0:1]
                    # Try creating conditions with `cond`; fall back if unsupported.
                    try:
                        conditions = self.network.module.preprocess_conditions(
                            clip_f, sync_f, text_f,
                            cond=unmasked_cond,
                            object_null_mask=None
                        )
                    except TypeError:
                        # Backward-compatibility fallback when the `cond` argument is unsupported.
                        conditions = self.network.module.preprocess_conditions(clip_f, sync_f, text_f)
                    empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])
                    cfg_ode_wrapper = lambda t, x: self.network.module.ode_wrapper(
                        t, x, conditions, empty_conditions, self.cfg_strength)
                        
                    x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
                    x1_hat = self.network.module.unnormalize(x1_hat)
                    mel = self.features.decode(x1_hat)
                    audio = self.features.vocode(mel).cpu()[0]
                    self.log.log_spectrogram('train', f'spec-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)
            except Exception as e:
                self.log.warning(f'Error in extra logging: {e}')
                if self.cfg.debug:
                    raise

        # Save network weights and checkpoint if needed
        save_copy = it in self.save_copy_iterations

        if (it % self.save_weights_interval == 0 and it != 0) or save_copy:
            self.save_weights(it)

        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    def _align_time(self, A: torch.Tensor, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Align time dimension between attention A:(B,O,T,P or T) and presence P:(B,O,T) by truncating to min T.
        Returns (A_aligned, P_aligned)."""
        if A.dim() == 4:
            T_A = A.size(2)
        elif A.dim() == 3:
            T_A = A.size(2)
        else:
            raise ValueError('Unexpected attention dims for alignment')
        T_P = P.size(2)
        T = min(T_A, T_P)
        if A.dim() == 4:
            A = A[:, :, :T, :]
        else:
            A = A[:, :, :T]
        P = P[:, :, :T]
        return A, P

    def _safe_nanmean(self, x: torch.Tensor, dim=None) -> torch.Tensor:
        """nanmean for torch tensors; if all-NaN, returns NaN."""
        if dim is None:
            return torch.nanmean(x)
        return torch.nanmean(x, dim=dim)

    def _log_object_condition_diagnostics(self, cond: dict, it: int):
        """Compute and log compile-safe diagnostics based on presence and patch attention.
        Logs:
          - diag/auc_present_absent
          - diag/coverage@{tau}
          - diag/temporal_var_present
          - diag/obj_entropy
        """
        if not isinstance(cond, dict):
            return
        if 'obj_patch_attn' not in cond or 'presence_mask' not in cond:
            return
        attn = cond['obj_patch_attn']  # (B, O, T, P)
        presence = cond['presence_mask']  # (B, O, T)
        if attn is None or presence is None:
            return
        # Reduce patches first on GPU to shrink data, then move to CPU for safety
        if attn.dim() != 4:
            return
        # max over patches -> (B, O, T)
        A = attn.max(dim=-1).values
        # Align time
        A, P = self._align_time(A, presence)
        P = (P > 0.5).to(A.dtype)

        # Present/absent masks and counts
        present_count = P.sum()
        absent_count = (1.0 - P).sum()
        # Avoid empty sets
        present_mean = (A * P).sum() / present_count.clamp(min=1)
        absent_mean = (A * (1.0 - P)).sum() / absent_count.clamp(min=1)
        auc_delta = (present_mean - absent_mean).item()
        self.log.log_scalar('diag/auc_present_absent', float(auc_delta), it)

        # Coverage@tau on present frames only
        for tau in self.diag_cov_thresholds:
            cov = ((A >= float(tau)).to(A.dtype) * P).sum() / present_count.clamp(min=1)
            self.log.log_scalar(f'diag/coverage@{tau}', float(cov.item()), it)

        # Temporal stability: variance over time on present frames (per (B,O), then average)
        # Compute per (B,O) masked mean and variance
        P_sum_BO = P.sum(dim=2)  # (B,O)
        # mean per (B,O)
        mean_BO = (A * P).sum(dim=2) / P_sum_BO.clamp(min=1)
        # E[x^2]
        ex2_BO = ((A * A) * P).sum(dim=2) / P_sum_BO.clamp(min=1)
        var_BO = (ex2_BO - mean_BO * mean_BO)
        # Only count entries with at least 2 present frames
        valid_BO = (P_sum_BO >= 2).to(A.dtype)
        temporal_var = (var_BO * valid_BO).sum() / valid_BO.sum().clamp(min=1)
        self.log.log_scalar('diag/temporal_var_present', float(temporal_var.item()), it)

        # Object competition: entropy over objects per frame H(b,t) and then mean
        # softmax over O for each (B,T)
        # reshape to (B, O, T) already
        A_bt = A.transpose(1, 2)  # (B, T, O)
        A_bt = A_bt - A_bt.amax(dim=2, keepdim=True)  # stabilize
        q = torch.softmax(A_bt, dim=2).clamp_min(1e-8)
        entropy = -(q * torch.log(q)).sum(dim=2)  # (B, T)
        obj_entropy = entropy.mean().item()
        self.log.log_scalar('diag/obj_entropy', float(obj_entropy), it)

        # Log delta gate scalars every time this diagnostic is called (e.g., with loss)
        try:
            # unwrap DDP if wrapped
            net = self.network.module if isinstance(self.network, DDP) else self.network
            stats = net.get_delta_gate_stats() if hasattr(net, "get_delta_gate_stats") else {}
        except Exception:
            stats = {}

        if stats:
            # TensorBoard
            if hasattr(self, "log") and self.log is not None:
                for k, v in stats.items():
                    # TensorboardLogger usually has add_scalar or log_scalar
                    if hasattr(self.log, "add_scalar"):
                        self.log.add_scalar(k, v, it)
                    elif hasattr(self.log, "log_scalar"):
                        self.log.log_scalar(k, v, it)

            # WandB (mirror)
            if (wandb is not None) and getattr(self.cfg.wandb, "enable", False) and wandb.run is not None:
                wandb.log(stats | {"_step": it}, step=it)

    def _infer_grid_hw(self, P: int) -> tuple[int, int] | None:
        """Infer a plausible (H,W) grid from patch count P. Prefer square; else closest factors."""
        g = int(math.isqrt(P))
        if g * g == P:
            return g, g
        # Find closest factorization
        for h in range(g, 0, -1):
            if P % h == 0:
                return h, P // h
        return None

    def _log_object_condition_visualization(self, cond: dict, it: int):
        """Log a lightweight attention heatmap grid for b=0 and one object across a few frames."""
        if 'obj_patch_attn' not in cond or 'presence_mask' not in cond:
            return
        attn = cond['obj_patch_attn']  # (B,O,T,P)
        presence = cond['presence_mask']  # (B,O,T)
        if attn is None or presence is None or attn.dim() != 4:
            return
        B, O, T, Pn = attn.shape
        if B == 0 or O == 0 or T == 0 or Pn == 0:
            return
        # Choose sample and object
        b0 = 0
        A = attn.max(dim=-1).values  # (B,O,T)
        A, Pmask = self._align_time(A, presence)
        # pick object with most presence, fallback to highest avg attention
        pres_len = Pmask[b0].sum(dim=1)  # (O,)
        if pres_len.max() > 0:
            o0 = int(torch.argmax(pres_len).item())
        else:
            o0 = int(torch.argmax(A[b0].mean(dim=1)).item())
        # select frames: top 8 present frames by attention, else top-8 overall
        a_b0_o0 = attn[b0, o0]  # (T,P)
        A_b0_o0 = A[b0, o0]    # (T)
        present_t = (Pmask[b0, o0] > 0.5)
        num_vis = min(8, A_b0_o0.numel())
        if present_t.any():
            idx = torch.topk(A_b0_o0[present_t], k=min(num_vis, int(present_t.sum().item())), largest=True).indices
            # map to absolute indices
            t_idx = torch.nonzero(present_t, as_tuple=False).squeeze(1)[idx]
        else:
            t_idx = torch.topk(A_b0_o0, k=num_vis, largest=True).indices
        # Infer grid
        hw = self._infer_grid_hw(a_b0_o0.size(1))
        if hw is None:
            return
        H, W = hw
        # Build a single-row grid image
        tiles = []
        for t in t_idx.tolist():
            heat = a_b0_o0[t]  # (P)
            heat = heat.detach().float()
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat = heat.view(H, W).cpu().numpy()
            # to uint8 grayscale
            tile = (heat * 255.0).astype(np.uint8)
            tiles.append(tile)
        if not tiles:
            return
        spacer = 4  # pixels between tiles
        h = H
        w = W * len(tiles) + spacer * (len(tiles) - 1)
        canvas = np.zeros((h, w), dtype=np.uint8)
        x = 0
        for tile in tiles:
            canvas[:, x:x + W] = tile
            x += W + spacer
        # Convert to 3-channel for better viewer support
        canvas_rgb = np.stack([canvas, canvas, canvas], axis=-1)
        self.log.log_image('train', f'oc_attn_b0_o{o0}', canvas_rgb, it)

    @torch.inference_mode()
    def validation_pass(self, data, it: int = 0):
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data['clip_features'].cuda(non_blocking=True)
            sync_f = data['sync_features'].cuda(non_blocking=True)
            text_f = data['text_features'].cuda(non_blocking=True)
            video_exist = data['video_exist'].cuda(non_blocking=True)
            text_exist = data['text_exist'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)
            a_std = data['a_std'].cuda(non_blocking=True)

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat
            a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
            x1 = a_mean + a_std * a_randn  # raw (unnormalized) GT posterior sample used for loss
            gt_latents = x1.detach()  # keep a copy (no grad) for preview logging

            # Extract object conditions.
            cond = self._extract_object_cond(data)
            try:
                cond['video_exist'] = video_exist
            except Exception:
                pass

            self.log.data_timer.end()
            loss, mean_loss, t = self.val_fn(clip_f.clone(), sync_f.clone(), text_f.clone(), x1, cond=cond)

            self.val_integrator.add_binned_tensor('binned_loss', loss, t)
            self.val_integrator.add_dict({'loss': mean_loss})

        self.log.data_timer.start()

        # Media previews are only produced from the first eval_loader batch through inference_pass.

    @torch.inference_mode()
    def inference_pass(self,
                       data,
                       it: int,
                       data_cfg: DictConfig,
                       *,
                       save_eval: bool = True,
                       is_first_eval_batch: bool = False,
                       force_preview: bool = False) -> Path:
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data['clip_features'].cuda(non_blocking=True)
            sync_f = data['sync_features'].cuda(non_blocking=True)
            text_f = data['text_features'].cuda(non_blocking=True)
            video_exist = data['video_exist'].cuda(non_blocking=True)
            text_exist = data['text_exist'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)  # for the shape only

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            # sample
            x0 = torch.empty_like(a_mean).normal_(generator=self.rng)
            # Extract object conditions.
            cond = self._extract_object_cond(data)
            # Try creating conditions with `cond`; fall back if unsupported.

            try:
                conditions = self.network.module.preprocess_conditions(
                    clip_f, sync_f, text_f,
                    cond=cond,
                    object_null_mask=None
                )
            except TypeError:
                conditions = self.network.module.preprocess_conditions(clip_f, sync_f, text_f)
            empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])
            cfg_ode_wrapper = lambda t, x: self.network.module.ode_wrapper(
                t, x, conditions, empty_conditions, self.cfg_strength)
                
            x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
            x1_hat = self.network.module.unnormalize(x1_hat)
            mel = self.features.decode(x1_hat)
            audio = self.features.vocode(mel).cpu()
            for i in range(audio.shape[0]):
                video_id = data['id'][i]
                # if (not self.for_training) and i == 0:
                #     # save very few videos
                    
                #     # self.test_video_joiner.join(video_id, f'{video_id}', audio[i].transpose(0, 1))
                #     try:
                #         self.test_video_joiner.join(video_id, f'{video_id}', audio[i].transpose(0, 1))
                #     except Exception as e:
                #         self.log.warning(f'Test video join failed: {e}')

                if data_cfg.output_subdir is not None:
                    # validation
                    if save_eval:
                        iter_naming = f'{it:09d}'
                    else:
                        iter_naming = 'val-cache'
                    audio_dir = self.log.log_audio(iter_naming,
                                                   f'{video_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate,
                                                   subdir=Path(data_cfg.output_subdir),
                                                   upload_wandb=False)
                    # if save_eval and i == 0:
                    #     # self.val_video_joiner.join(video_id, f'{iter_naming}-{video_id}',
                    #     #                            audio[i].transpose(0, 1))
                    #     try:
                    #         self.val_video_joiner.join(video_id, f'{iter_naming}-{video_id}',
                    #                                    audio[i].transpose(0, 1))
                    #     except Exception as e:
                    #         self.log.warning(f'Video join failed: {e}')

                else:
                    # full test set, usually
                    audio_dir = self.log.log_audio(f'{data_cfg.tag}-sampled',
                                                   f'{video_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate,
                                                   upload_wandb=False)

            # ----------------------------
            # W&B media preview: only on the first eval_loader batch and only at eval intervals.
            # ----------------------------
            try:
                wb_cfg = getattr(self.cfg, 'wandb', None)
                eval_interval = int(getattr(self.cfg, 'eval_interval', 1000))
                if (wb_cfg is not None
                    and getattr(wb_cfg, 'val_preview_enable', True)
                    and hasattr(self.log, 'wandb_enabled')
                    and self.log.wandb_enabled
                    and local_rank == 0
                    and is_first_eval_batch
                    and (force_preview or ((it + 1) % eval_interval == 0))):
                    import wandb
                    max_preview = int(getattr(wb_cfg, 'val_preview_count', 2))
                    B = a_mean.shape[0]
                    n = min(B, max_preview)
                    # Reuse the GT mel/audio tensors created above.
                    mel_gt = mel[:n]
                    audio_gt = audio[:n]
                    # Predict a prior sample.
                    with torch.no_grad():
                        x0 = torch.empty_like(a_mean[:n]).normal_(generator=self.rng)
                        clip_sub = clip_f[:n].clone()
                        sync_sub = sync_f[:n].clone()
                        text_sub = text_f[:n].clone()
                        cond_sub = self._extract_object_cond({k: (v[:n] if hasattr(v, '__getitem__') else v)
                                                              for k, v in data.items() if k in (
                                                                  'velocity_f','mass_f','presence_mask','clip_patches','obj_patch_attn','obj_empty_mask','velocity_missing','obj_label_ids','obj_prompt_ids')})
                        try:
                            cond_sub['video_exist'] = video_exist[:n]
                        except Exception:
                            pass
                        try:
                            conditions = self.network.module.preprocess_conditions(
                                clip_sub, sync_sub, text_sub, cond=cond_sub, object_null_mask=None)
                        except TypeError:
                            conditions = self.network.module.preprocess_conditions(clip_sub, sync_sub, text_sub)
                        empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])
                        cfg_ode_wrapper = lambda t_, x_: self.network.module.ode_wrapper(
                            t_, x_, conditions, empty_conditions, self.cfg_strength)
                        x1_pred = self.fm.to_data(cfg_ode_wrapper, x0)
                        x1_pred = self.network.module.unnormalize(x1_pred)
                        mel_pred = self.features.decode(x1_pred)
                        audio_pred = self.features.vocode(mel_pred).cpu()
                    for j in range(n):
                        vid = data.get('id', [f'sample{j}'])[j]
                        step = it
                        try:
                            def _prep(a):
                                t = a.detach().cpu().float()
                                if t.ndim == 2 and t.shape[1] < t.shape[0]:
                                    t = t.transpose(0,1)
                                if t.ndim == 2 and t.shape[0] == 1:
                                    t = t[0]
                                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                                t = t.clamp(-1.0,1.0).contiguous()
                                return t.numpy()
                            wandb.log({
                                f'val_preview/{vid}/audio_gt': wandb.Audio(_prep(audio_gt[j]), sample_rate=self.sample_rate),
                                f'val_preview/{vid}/audio_pred': wandb.Audio(_prep(audio_pred[j]), sample_rate=self.sample_rate)
                            }, step=step)
                        except Exception as e:
                            self.log.warning(f'wandb audio preview log failed ({vid}): {e}')
                        try:
                            import matplotlib.pyplot as _plt
                            def _spec_img(m):
                                fig, ax = _plt.subplots(figsize=(4,2))
                                ax.imshow(m[j].cpu().float().numpy(), origin='lower', aspect='auto', interpolation='nearest')
                                ax.axis('off')
                                return fig
                            fig_gt = _spec_img(mel_gt)
                            fig_pred = _spec_img(mel_pred)
                            wandb.log({
                                f'val_preview/{vid}/spec_gt': wandb.Image(fig_gt),
                                f'val_preview/{vid}/spec_pred': wandb.Image(fig_pred)
                            }, step=step)
                            _plt.close(fig_gt); _plt.close(fig_pred)
                        except Exception as e:
                            self.log.warning(f'wandb spectrogram preview log failed ({vid}): {e}')
                        try:
                            video_paths = data.get('video_path', None)
                            if video_paths is not None:
                                vpath = video_paths[j]
                                if isinstance(vpath, str) and os.path.exists(vpath):
                                    wandb.log({f'val_preview/{vid}/video': wandb.Video(vpath, fps=8, format='mp4')}, step=step)
                        except Exception as e:
                            self.log.warning(f'wandb video preview log failed ({vid}): {e}')
            except Exception as e:
                if self.cfg.debug:
                    raise
                self.log.warning(f'Inference media preview failed: {e}')

        return Path(audio_dir)

    # @torch.inference_mode()
    # def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
    #     with torch.amp.autocast('cuda', enabled=False):
    #         if local_rank == 0:
    #             extract(audio_path=audio_dir,
    #                     output_path=audio_dir / 'cache',
    #                     device='cuda',
    #                     batch_size=32,
    #                     audio_length=8)
    #             output_metrics = evaluate(gt_audio_cache=Path(data_cfg.gt_cache),
    #                                       pred_audio_cache=audio_dir / 'cache')
    #             for k, v in output_metrics.items():
    #                 # pad k to 10 characters
    #                 # pad v to 10 decimal places
    #                 self.log.log_scalar(f'{data_cfg.tag}/{k}', v, it)
    #                 self.log.info(f'{data_cfg.tag}/{k:<10}: {v:.10f}')
    #         else:
    #             output_metrics = None

    #     return output_metrics

    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        with torch.amp.autocast('cuda', enabled=False):
            if local_rank == 0:
                from av_bench.evaluate import evaluate
                from av_bench.extract import extract

                pred_cache = audio_dir / 'cache'
                pred_cache.mkdir(parents=True, exist_ok=True)
                extract(
                    audio_path=audio_dir,
                    output_path=pred_cache,
                    device='cuda',
                    batch_size=32,
                    audio_length=float(getattr(self, 'duration_sec', 8.0)),
                )
                if not getattr(data_cfg, 'compute_metrics', True):
                    self.log.info(f'{data_cfg.tag}: prediction cache extracted; metric computation skipped')
                    return None
                output_metrics = evaluate(
                    gt_audio_cache=Path(data_cfg.gt_cache),
                    pred_audio_cache=pred_cache
                )
                for k, v in output_metrics.items():
                    self.log.log_scalar(f'{data_cfg.tag}/{k}', v, it)  # Also reaches W&B when configured.
                    self.log.info(f'{data_cfg.tag}/{k:<18}: {v:.6f}')
            else:
                output_metrics = None
        return output_metrics

    @torch.inference_mode()
    def evaluate_dataset(self,
                         loader,
                         data_cfg: DictConfig,
                         *,
                         it: int = 0,
                         force_preview: bool = False) -> dict[str, float] | None:
        audio_dir: Path | None = None
        total = len(loader) if hasattr(loader, "__len__") else None
        is_rank0 = (local_rank == 0)
        pbar = tqdm(total=total, desc=f"{data_cfg.tag} generate", disable=not is_rank0)

        t0 = time.time()
        for bi, batch in enumerate(loader):
            audio_dir = self.inference_pass(
                batch,
                it=it,
                data_cfg=data_cfg,
                save_eval=True,
                is_first_eval_batch=(bi == 0),
                force_preview=force_preview,
            )
            if is_rank0:
                pbar.update(1)
        if is_rank0:
            pbar.close()
            self.log.info(f'[{data_cfg.tag}] generation finished in {time.time()-t0:.1f}s; waiting for ranks...')

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if is_rank0 and audio_dir is not None:
            self.log.info(f'[{data_cfg.tag}] extracting + evaluating: dir={audio_dir}')
            return self.eval(audio_dir, it, data_cfg)
        return None

    def save_weights(self, it, save_copy=False):
        if local_rank != 0:
            return

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_{it}.pth'
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f'Network weights saved to {model_path}.')

        # if last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)
            self.log.info(f'Network weights shadowed to {shadow_path}.')

        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f'Network weights saved to {model_path}.')

    def save_checkpoint(self, it, save_copy=False):
        if local_rank != 0:
            return

        checkpoint = {
            'it': it,
            'weights': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_ckpt_{it}.pth'
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        # if ckpt_last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f'Checkpoint shadowed to {shadow_path}.')

        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if not ckpt_path.exists():
            info_if_rank_zero(self.log, f'No checkpoint found at {ckpt_path}.')
            return None
        return ckpt_path

    def get_latest_weight_path(self):
        weight_path = self.run_path / f'{self.exp_id}_last.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def get_final_ema_weight_path(self):
        weight_path = self.run_path / f'{self.exp_id}_ema_final.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        it = checkpoint['it']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            self.log.info(f'EMA states loaded from step {self.ema.step}')

        map_location = 'cuda:%d' % local_rank
        self.network.module.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info(f'Global iteration {it} loaded.')
        self.log.info('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_weights_in_memory(self, src_dict):
        self.network.module.load_weights(src_dict)
        self.log.info('Network weights loaded from memory.')

    def load_weights(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        self.log.info(f'Importing network weights from {path}...')
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self
