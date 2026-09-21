"""
Dumps things to tensorboard and console
"""

import datetime
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import soundfile as sf
from PIL import Image
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    _WANDB_AVAILABLE = False

from core.utils.email_utils import EmailSender
from core.utils.time_estimator import PartialTimeEstimator, TimeEstimator
from core.utils.timezone import my_timezone


def tensor_to_numpy(image: torch.Tensor):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np


def detach_to_cpu(x: torch.Tensor):
    return x.detach().cpu()


def fix_width_trunc(x: float):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))


def plot_spectrogram(spectrogram: np.ndarray, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(spectrogram, origin="lower", aspect="auto", interpolation="nearest")


class TensorboardLogger:

    def __init__(self,
                 exp_id: str,
                 run_dir: Union[Path, str],
                 py_logger: logging.Logger,
                 *,
                 is_rank0: bool = False,
                 enable_email: bool = False,
                 wandb_cfg: Optional[dict] = None):
        self.exp_id = exp_id
        self.run_dir = Path(run_dir)
        self.py_log = py_logger
        self.email_sender = EmailSender(exp_id, enable=(is_rank0 and enable_email))
        if is_rank0:
            self.tb_log = SummaryWriter(run_dir)
        else:
            self.tb_log = None

        # wandb state
        self.wandb_enabled = False
        self._wandb_log_audio: bool = False
        self._wandb_flush_interval: int = int(wandb_cfg.get('flush_interval', 1)) if wandb_cfg else 1
        self._wandb_buffer: dict = {}
        if is_rank0 and wandb_cfg is not None and wandb_cfg.get('enable', False):
            if not _WANDB_AVAILABLE:
                self.py_log.warning('wandb requested but not installed. Skipping wandb init.')
            else:
                self._init_wandb(exp_id, run_dir, wandb_cfg)

        # Get current git info for logging
        try:
            import git
            repo = git.Repo(".")
            git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
        except Exception:
            self.py_log.warning('Failed to fetch git info. Defaulting to None')
            git_info = 'None'

        self.log_string('git', git_info)

        # log the SLURM job id if available
        job_id = os.environ.get('SLURM_JOB_ID', None)
        if job_id is not None:
            self.log_string('slurm_job_id', job_id)
            self.email_sender.send(f'Job {job_id} started', f'Job started {run_dir}')

        # used when logging metrics
        self.batch_timer: TimeEstimator = None
        self.data_timer: PartialTimeEstimator = None

        self.nan_count = defaultdict(int)

    def log_scalar(self, tag: str, x: float, it: int):
        if self.tb_log is None and not self.wandb_enabled:
            return
        if math.isnan(x) and 'grad_norm' not in tag:
            self.nan_count[tag] += 1
            if self.nan_count[tag] == 10:
                self.email_sender.send(
                    f'Nan detected in {tag} @ {self.run_dir}',
                    f'Nan detected in {tag} at iteration {it}; run_dir: {self.run_dir}')
        else:
            self.nan_count[tag] = 0
        if self.tb_log is not None:
            self.tb_log.add_scalar(tag, x, it)
        if self.wandb_enabled:
            # buffer scalars then flush every _wandb_flush_interval
            self._wandb_buffer[tag] = x
            # Always keep global_step synchronized with latest it
            self._wandb_buffer['global_step'] = it
            # if (it % self._wandb_flush_interval) == 0:
            if (it % self._wandb_flush_interval == 0 or it == 0):
                try:
                    wandb.log(self._wandb_buffer, step=it)
                except Exception as e:
                    self.py_log.warning(f'wandb flush failed: {e}')
                finally:
                    self._wandb_buffer = {}

    def log_metrics(self,
                    prefix: str,
                    metrics: dict[str, float],
                    it: int,
                    ignore_timer: bool = False):
        msg = f'{self.exp_id}-{prefix} - it {it:6d}: '
        metrics_msg = ''
        for k, v in sorted(metrics.items()):
            self.log_scalar(f'{prefix}/{k}', v, it)
            metrics_msg += f'{k: >10}:{v:.7f},\t'

        if self.batch_timer is not None and not ignore_timer:
            self.batch_timer.update()
            avg_time = self.batch_timer.get_and_reset_avg_time()
            data_time = self.data_timer.get_and_reset_avg_time()

            # add time to tensorboard
            self.log_scalar(f'{prefix}/avg_time', avg_time, it)
            self.log_scalar(f'{prefix}/data_time', data_time, it)

            est = self.batch_timer.get_est_remaining(it)
            est = datetime.timedelta(seconds=est)
            if est.days > 0:
                remaining_str = f'{est.days}d {est.seconds // 3600}h'
            else:
                remaining_str = f'{est.seconds // 3600}h {(est.seconds%3600) // 60}m'
            eta = datetime.datetime.now(timezone(my_timezone)) + est
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S %Z%z')
            time_msg = f'avg_time:{avg_time:.3f},data:{data_time:.3f},remaining:{remaining_str},eta:{eta_str},\t'
            msg = f'{msg} {time_msg}'

        msg = f'{msg} {metrics_msg}'
        self.py_log.info(msg)

    def log_histogram(self, tag: str, hist: torch.Tensor, it: int):
        if self.tb_log is None and not self.wandb_enabled:
            return
        # hist should be a 1D tensor
        hist = hist.cpu().numpy()
        fig, ax = plt.subplots()
        x_range = np.linspace(0, 1, len(hist))
        ax.bar(x_range, hist, width=1 / (len(hist) - 1))
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
        plt.tight_layout()
        if self.tb_log is not None:
            self.tb_log.add_figure(tag, fig, it)
        if self.wandb_enabled:
            wandb.log({tag: wandb.Image(fig), 'global_step': it}, step=it)
        plt.close()

    def log_image(self, prefix: str, tag: str, image: np.ndarray, it: int):
        image_dir = self.run_dir / f'{prefix}_images'
        image_dir.mkdir(exist_ok=True, parents=True)

        image = Image.fromarray(image)
        image.save(image_dir / f'{it:09d}_{tag}.png')
        if self.wandb_enabled:
            wandb.log({f'{prefix}/{tag}': wandb.Image(np.array(image)), 'global_step': it}, step=it)

    def log_audio(self,
                  prefix: str,
                  tag: str,
                  waveform: torch.Tensor,
                  it: Optional[int] = None,
                  *,
                  subdir: Optional[Path] = None,
                  sample_rate: int = 16000,
                  upload_wandb: Optional[bool] = None) -> Path:
        if subdir is None:
            audio_dir = self.run_dir / prefix
        else:
            audio_dir = self.run_dir / subdir / prefix
        audio_dir.mkdir(exist_ok=True, parents=True)

        if it is None:
            name = f'{tag}.flac'
        else:
            name = f'{it:09d}_{tag}.flac'

        audio_path = audio_dir / name
        audio_tensor = waveform.cpu().float()
        try:
            torchaudio.save(audio_path,
                            audio_tensor,
                            sample_rate=sample_rate,
                            channels_first=True)
        except Exception as exc:
            self.py_log.warning(f'torchaudio.save failed for {audio_path}, falling back to soundfile: {exc}')
            sf.write(str(audio_path),
                     audio_tensor.transpose(0, 1).contiguous().numpy(),
                     sample_rate,
                     format='FLAC')
        effective_upload = self._wandb_log_audio if upload_wandb is None else upload_wandb
        if self.wandb_enabled and effective_upload:
            try:
                audio_t = waveform.detach().cpu().float()
                # Remove any NaN/Inf (replace with 0)
                if torch.isnan(audio_t).any() or torch.isinf(audio_t).any():
                    audio_t = torch.nan_to_num(audio_t, nan=0.0, posinf=0.0, neginf=0.0)
                # Acceptable shapes for wandb: (T,) or (C, T)
                # Common shapes we might see: (T,), (C, T), (T, C)
                if audio_t.ndim == 2 and audio_t.shape[1] < audio_t.shape[0]:
                    # (T, C) -> transpose to (C, T)
                    audio_t = audio_t.transpose(0, 1)
                if audio_t.ndim == 2 and audio_t.shape[0] == 1:
                    # collapse mono to (T,)
                    audio_t = audio_t[0]
                # Clamp to [-1,1]
                audio_t = audio_t.clamp(-1.0, 1.0).contiguous()
                audio_np = audio_t.numpy()
                wandb.log({
                    f'{prefix}/{tag}': wandb.Audio(audio_np, sample_rate=sample_rate),
                    'global_step': (it if it is not None else 0)
                }, step=(it if it is not None else None))
            except Exception as e:
                self.py_log.warning(f'wandb audio log failed: {e}')
        return Path(audio_dir)

    def log_spectrogram(
        self,
        prefix: str,
        tag: str,
        spec: torch.Tensor,
        it: Optional[int],
        *,
        subdir: Optional[Path] = None,
    ):
        if subdir is None:
            spec_dir = self.run_dir / prefix
        else:
            spec_dir = self.run_dir / subdir / prefix
        spec_dir.mkdir(exist_ok=True, parents=True)

        if it is None:
            name = f'{tag}.png'
        else:
            name = f'{it:09d}_{tag}.png'

        plot_spectrogram(spec.cpu().float())
        plt.tight_layout()
        plt.savefig(spec_dir / name)
        if self.wandb_enabled:
            try:
                fig, ax = plt.subplots(1, 1)
                plot_spectrogram(spec.cpu().float().numpy(), ax=ax)
                wandb.log({f'{prefix}/{tag}': wandb.Image(fig), 'global_step': (it if it is not None else 0)}, step=(it if it is not None else None))
                plt.close(fig)
            except Exception as e:
                self.py_log.warning(f'wandb spectrogram log failed: {e}')
        plt.close()

    def log_string(self, tag: str, x: str):
        self.py_log.info(f'{tag} - {x}')
        if self.tb_log is not None:
            self.tb_log.add_text(tag, x)
        if self.wandb_enabled:
            wandb.log({f'string/{tag}': x})

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)

        self.email_sender.send(f'Error occurred in {self.run_dir}', x)

    def complete(self):
        self.email_sender.send(f'Job completed in {self.run_dir}', 'Job completed')
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass

    # ------------------------------
    # Internal helpers (wandb)
    # ------------------------------
    def _init_wandb(self, exp_id: str, run_dir: Union[str, Path], wandb_cfg: dict):
        try:
            # Optional custom temp/cache dir for wandb media (avoid /tmp permission issues on some clusters)
            custom_tmp = wandb_cfg.get('tmp_dir', None)
            if custom_tmp:
                try:
                    tmp_path = Path(custom_tmp).expanduser().resolve()
                    tmp_path.mkdir(parents=True, exist_ok=True)
                    # Set a few env vars wandb / tempfile might respect
                    os.environ.setdefault('WANDB_TMP_DIR', str(tmp_path))
                    os.environ.setdefault('TMPDIR', str(tmp_path))
                    os.environ.setdefault('TEMP', str(tmp_path))
                    os.environ.setdefault('TMP', str(tmp_path))
                except Exception as _e:
                    self.py_log.warning(f'Failed to prepare custom wandb tmp_dir={custom_tmp}: {_e}')
            settings = dict(project=wandb_cfg.get('project', 'core'),
                            name=wandb_cfg.get('name', exp_id),
                            entity=wandb_cfg.get('entity') or None,
                            group=wandb_cfg.get('group') or None,
                            tags=wandb_cfg.get('tags') or [],
                            mode=wandb_cfg.get('mode', 'online'),
                            dir=str(run_dir),
                            id=wandb_cfg.get('id') or None,
                            resume=wandb_cfg.get('resume') or None,)
            cfg_upload_mode = wandb_cfg.get('config_upload', 'minimal')
            config_payload = {}
            if cfg_upload_mode == 'full':
                # Hydra cfg will be passed later by caller if desired; keep placeholder
                config_payload['exp_id'] = exp_id
            else:
                config_payload['exp_id'] = exp_id
            wandb.init(**settings, config=config_payload, reinit=False)
            self.wandb_enabled = True
            self._wandb_log_audio = bool(wandb_cfg.get('log_audio', False))
            if bool(wandb_cfg.get('watch_model', False)):
                try:
                    wandb.watch_called = False  # reset safeguard
                except Exception:
                    pass
            self.py_log.info('wandb initialized.')
        except Exception as e:
            self.py_log.warning(f'Failed to initialize wandb: {e}')
            self.wandb_enabled = False
