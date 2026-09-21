import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from core.ext.rotary_embeddings import compute_rope_rotations
from core.model.embeddings import TimestepEmbedder
from core.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from core.model.transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()

# ---- Lightweight debug counters for on-demand OC logging ----
_OC_FW_LOG_COUNTER = 0  # module-level counter for framewise object feature logging
_OC_CONDITION_FAILURE_COUNTS = {}


def _warn_object_condition_failure(reason: str, exc: Exception) -> None:
    count = _OC_CONDITION_FAILURE_COUNTS.get(reason, 0) + 1
    _OC_CONDITION_FAILURE_COUNTS[reason] = count
    if count <= 3 or count in (10, 100, 1000) or count % 10000 == 0:
        log.warning('Condition input diagnostic: %s (process count=%d): %s', reason, count, exc)


# Fourier encoder for scalar physical inputs.
class ScalarFourierMLP(nn.Module):
    """
    Input: (..., 1) scalar
    Output: (..., Dh) via [Fourier expand (2K or 2K+1 or 2K+2)] -> LN -> MLP
    """
    def __init__(self, hidden_dim: int, K: int = 16, freq_max: float = 128.0,
                 mode: str = "2K+2", dc_free_cos: bool = True):
        super().__init__()
        assert K >= 1, "K must be >= 1"
        assert mode in ("2K", "2K+1", "2K+2"), f"unsupported mode: {mode}"
        print("Fourier mode: ", mode)
        self.K = int(K)
        self.freq_max = float(freq_max)
        self.mode = mode
        self.dc_free_cos = bool(dc_free_cos)

        # logspace frequencies in [1, freq_max]
        # shape: (K,)
        omegas = torch.logspace(start=0.0, end=torch.log10(torch.tensor(self.freq_max)), steps=self.K)
        self.register_buffer("omegas", omegas, persistent=False)

        # feature dim after concat
        base = 2 * self.K  # sin, cos
        if self.mode == "2K+1":
            in_dim = base + 1  # + raw
        elif self.mode == "2K+2":
            in_dim = base + 2  # + raw, + |raw|
        else:
            in_dim = base      # 2K

        # small projection head: LN -> Linear -> SiLU -> Linear -> Dh
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 1)
        returns: (..., Dh)
        """
        # (..., 1) -> broadcast with omegas (K,)
        # angle = 2π * ω * x
        angle = 2.0 * torch.pi * x * self.omegas.view(*(1 for _ in range(x.ndim - 1)), -1)  # (..., K)

        s = torch.sin(angle)  # (..., K)
        c = torch.cos(angle)  # (..., K)
        if self.dc_free_cos:
            c = c - 1.0

        feats = [s, c]  # 2K
        if self.mode == "2K+1":
            feats.append(x[..., 0:1])              # raw
        elif self.mode == "2K+2":
            feats.append(x[..., 0:1])              # raw
            feats.append(x[..., 0:1].abs())        # |raw|

        f = torch.cat(feats, dim=-1)  # (..., in_dim)
        f = self.ln(f)
        return self.proj(f)
# -------------------------------------------------------------------------------


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor
    # Object conditions.
    obj_frame_mass: Optional[torch.Tensor] = None   # (B, T_latent_or_obj, Dh)
    obj_frame_vel: Optional[torch.Tensor] = None    # (B, T_latent_or_obj, Dh)
    obj_global_c: Optional[torch.Tensor] = None     # (B, Dh)

# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 v2: bool = False,
                 object_cond_cfg: Optional[dict] = None,
                 architecture_debug: bool = False,
                 use_delta_mod: bool = False,
                 multiply_mass_vel: bool = False
                 ) -> None:
        super().__init__()
        self.arch_debug = bool(architecture_debug)
        self.use_delta_mod = bool(use_delta_mod)
        self.multiply_mass_vel = bool(multiply_mass_vel)
        # Adapter for p = m ⊙ v.
        self.p_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.p_gain = nn.Parameter(torch.tensor(0.1))  # small initial gain

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )
        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       pre_only=(i == depth - fused_depth - 1),
                       use_delta_mod_latent=self.use_delta_mod) for i in range(depth - fused_depth)
        ])

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1,
                             use_delta_mod=self.use_delta_mod)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)

        self.initialize_weights()
        self.initialize_rotations()

        # Project ViT-B/16 patch tokens into the object-conditioning space.
        self.obj_cond_in = nn.Linear(sync_dim, hidden_dim)

        # Frame-aligned object-conditioning modules.
        self.patch_to_hidden = nn.Linear(sync_dim, hidden_dim)
        self.mass_embed = nn.Module
        self.velocity_embed = nn.Module

        self.mass_to_affine = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.vel_to_affine = nn.Linear(hidden_dim, 2 * hidden_dim)

        # self.obj_fuse_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.mass_fuse_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.vel_fuse_norm  = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # frame token: (1,1,Dh) broadcast to (B,T,Dh)
        self.empty_obj_frame_mass = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.empty_obj_frame_vel = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # global token: (1,Dh) broadcast to (B,Dh)
        self.empty_obj_global = nn.Parameter(torch.zeros(1, hidden_dim))
        
        nn.init.normal_(self.empty_obj_frame_mass, std=0.02)
        nn.init.normal_(self.empty_obj_frame_vel, std=0.02)
        nn.init.normal_(self.empty_obj_global, std=0.02)

        # per-frame velocity occlusion token (for frames where object present but speed missing)
        self.vel_occ_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.vel_occ_token, std=0.02)

        # object patch-level occlusion token (for frames where object is absent/occluded)
        # self.obj_occ_token = nn.Parameter(torch.zeros(sync_dim))
        # nn.init.normal_(self.obj_occ_token, std=0.02)

        self.obj_occ_token_h = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.obj_occ_token_h, std=0.02)

        # Object-conditioning configuration.
        oc = {}
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
            if isinstance(object_cond_cfg, DictConfig):
                oc = OmegaConf.to_container(object_cond_cfg, resolve=True) or {}
            elif isinstance(object_cond_cfg, dict):
                oc = object_cond_cfg
        except Exception:
            if isinstance(object_cond_cfg, dict):
                oc = object_cond_cfg

        # Optionally load normalization stats from an external JSON (produced by compute_object_condition_stats.py)
        try:
            if isinstance(oc, dict) and oc.get('use_stats_json', True):
                stats_path = oc.get('stats_json', None)
                if stats_path is not None:
                    from pathlib import Path as _P
                    sp = _P(stats_path)
                    if sp.is_file():
                        import json as _json
                        with sp.open('r') as _fh:
                            _stats_json = _json.load(_fh)
                        # mass_norm
                        if 'mass' in _stats_json and isinstance(_stats_json['mass'], dict):
                            oc.setdefault('mass_norm', {})
                            oc['mass_norm']['mean'] = float(_stats_json['mass'].get('mean', oc['mass_norm'].get('mean', 0.0)))
                            oc['mass_norm']['std'] = float(_stats_json['mass'].get('std', oc['mass_norm'].get('std', 1.0)))
                            # log stats
                            oc.setdefault('mass_log_norm', {})
                            oc['mass_log_norm']['mean'] = float(_stats_json['mass'].get('log_mean', _stats_json.get('mass_log_mean', oc['mass_log_norm'].get('mean', 0.0))))
                            oc['mass_log_norm']['std'] = float(_stats_json['mass'].get('log_std', _stats_json.get('mass_log_std', oc['mass_log_norm'].get('std', 1.0))))
                        # velocity
                        if 'velocity' in _stats_json and isinstance(_stats_json['velocity'], dict):
                            oc.setdefault('vel_norm', {})
                            oc['vel_norm']['mean'] = float(_stats_json['velocity'].get('mean', oc['vel_norm'].get('mean', 0.0)))
                            oc['vel_norm']['std'] = float(_stats_json['velocity'].get('std', oc['vel_norm'].get('std', 1.0)))
                            oc.setdefault('vel_log_norm', {})
                            oc['vel_log_norm']['mean'] = float(_stats_json['velocity'].get('log_mean', _stats_json.get('velocity_log_mean', oc['vel_log_norm'].get('mean', 0.0))))
                            oc['vel_log_norm']['std'] = float(_stats_json['velocity'].get('log_std', _stats_json.get('velocity_log_std', oc['vel_log_norm'].get('std', 1.0))))
                        if self.arch_debug and self._is_main_process():
                            try:
                                print(f"[OC][stats_json] loaded stats from {sp}", flush=True)
                            except Exception:
                                pass
        except Exception:
            if self.arch_debug and self._is_main_process():
                try:
                    print('[OC][stats_json] failed to load stats_json', flush=True)
                except Exception:
                    pass

        patch_mod = oc.get('patch_mod', {}) if isinstance(oc, dict) else {}
        obj_prune = oc.get('obj_prune', {}) if isinstance(oc, dict) else {}
        self.oc_lambda_mass = float(oc.get('lambda_mass', 1.0)) if isinstance(oc, dict) else 1.0
        self.oc_lambda_vel = float(oc.get('lambda_vel', 1.0)) if isinstance(oc, dict) else 1.0
        self.oc_pre_agg = bool(patch_mod.get('pre_agg', False)) if isinstance(patch_mod, dict) else False
        # Removed legacy per-patch (oc_top_k) and per-frame (oc_obj_top_m) pruning in favor of global pre-collate pruning.
        self.oc_top_k = None  # legacy placeholder, no longer used
        self.oc_obj_top_m = None  # legacy placeholder, no longer used
        # Gate for optional global object conditioning.
        self.oc_use_global = bool(oc.get('use_global', False)) if isinstance(oc, dict) else False
        # velocity object pooling mode: 'mean' (legacy presence-weighted mean) or 'gate'
        self.oc_vel_pool_mode = str(oc.get('vel_pool', 'mean')) if isinstance(oc, dict) else 'mean'
        self.oc_mass_pool_mode = str(oc.get('mass_pool', 'mean')) if isinstance(oc, dict) else 'mean'
        # Override from configuration when provided.
        self.multiply_mass_vel = bool(oc.get('multiply_mass_vel', self.multiply_mass_vel))
        # offline normalization config & stats
        norm_cfg = oc.get('norm', {}) if isinstance(oc, dict) else {}
        self.oc_norm_enable = bool(norm_cfg.get('enable', True)) if isinstance(norm_cfg, dict) else True
        self.oc_norm_eps = float(norm_cfg.get('epsilon', 1e-6)) if isinstance(norm_cfg, dict) else 1e-6
        self.oc_norm_clip = norm_cfg.get('clip', None) if isinstance(norm_cfg, dict) else None
        mass_norm = oc.get('mass_norm', {}) if isinstance(oc, dict) else {}
        vel_norm = oc.get('vel_norm', {}) if isinstance(oc, dict) else {}
        # optional log-domain stats (added)
        mass_log_norm = oc.get('mass_log_norm', {}) if isinstance(oc, dict) else {}
        vel_log_norm = oc.get('vel_log_norm', {}) if isinstance(oc, dict) else {}
        try:
            self.register_buffer('oc_mass_mean', torch.tensor(float(mass_norm.get('mean', 0.0)), dtype=torch.float32))
            self.register_buffer('oc_mass_std', torch.tensor(float(mass_norm.get('std', 1.0)), dtype=torch.float32))
        except Exception:
            self.register_buffer('oc_mass_mean', torch.zeros(1))
            self.register_buffer('oc_mass_std', torch.ones(1))
        try:
            self.register_buffer('oc_vel_mean', torch.tensor(float(vel_norm.get('mean', 0.0)), dtype=torch.float32))
            self.register_buffer('oc_vel_std', torch.tensor(float(vel_norm.get('std', 1.0)), dtype=torch.float32))
        except Exception:
            self.register_buffer('oc_vel_mean', torch.zeros(1))
            self.register_buffer('oc_vel_std', torch.ones(1))
        # log stats buffers (mass / velocity)
        try:
            self.register_buffer('oc_mass_log_mean', torch.tensor(float(mass_log_norm.get('mean', mass_log_norm.get('log_mean', 0.0))), dtype=torch.float32))
            self.register_buffer('oc_mass_log_std', torch.tensor(float(mass_log_norm.get('std', mass_log_norm.get('log_std', 1.0))), dtype=torch.float32))
        except Exception:
            self.register_buffer('oc_mass_log_mean', torch.zeros(1))
            self.register_buffer('oc_mass_log_std', torch.ones(1))
        try:
            self.register_buffer('oc_vel_log_mean', torch.tensor(float(vel_log_norm.get('mean', vel_log_norm.get('log_mean', 0.0))), dtype=torch.float32))
            self.register_buffer('oc_vel_log_std', torch.tensor(float(vel_log_norm.get('std', vel_log_norm.get('log_std', 1.0))), dtype=torch.float32))
        except Exception:
            self.register_buffer('oc_vel_log_mean', torch.zeros(1))
            self.register_buffer('oc_vel_log_std', torch.ones(1))
        self._oc_norm_logged = False
        # normalization feature toggles
        norm_mass_cfg = norm_cfg.get('mass', {}) if isinstance(norm_cfg, dict) else {}
        norm_vel_cfg = norm_cfg.get('velocity', {}) if isinstance(norm_cfg, dict) else {}
        self.oc_mass_use_z = bool(norm_mass_cfg.get('use_z', True))
        self.oc_mass_use_log1p = bool(norm_mass_cfg.get('use_log1p', False))
        self.oc_vel_use_z = bool(norm_vel_cfg.get('use_z', True))
        self.oc_vel_use_log1p = bool(norm_vel_cfg.get('use_log1p', False))
        # pre LayerNorm on object hidden features BEFORE FiLM (config: norm.pre_obj_layernorm)
        self.oc_pre_obj_layernorm = bool(norm_cfg.get('pre_obj_layernorm', True)) if isinstance(norm_cfg, dict) else True
        if self.oc_pre_obj_layernorm:
            self.oc_obj_pre_ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        else:
            self.oc_obj_pre_ln = None
        # temporal smoothing config (depthwise Conv1d residual per branch)
        ts_cfg = oc.get('temporal_smooth', {}) if isinstance(oc, dict) else {}
        self.oc_ts_enable = bool(ts_cfg.get('enable', False))
        self.oc_ts_kernel = int(ts_cfg.get('kernel', 5)) if self.oc_ts_enable else 0
        self.oc_ts_mass = bool(ts_cfg.get('mass', True)) if self.oc_ts_enable else False
        self.oc_ts_vel = bool(ts_cfg.get('velocity', True)) if self.oc_ts_enable else False
        if self.oc_ts_enable:
            k = self.oc_ts_kernel if self.oc_ts_kernel % 2 == 1 else (self.oc_ts_kernel + 1)
            pad = k // 2
            if self.oc_ts_mass:
                self.mass_temporal_smooth = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad, groups=hidden_dim, bias=False)
                nn.init.zeros_(self.mass_temporal_smooth.weight)
            else:
                self.mass_temporal_smooth = None
            if self.oc_ts_vel:
                self.vel_temporal_smooth = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=pad, groups=hidden_dim, bias=False)
                nn.init.zeros_(self.vel_temporal_smooth.weight)
            else:
                self.vel_temporal_smooth = None
        else:
            self.mass_temporal_smooth = None
            self.vel_temporal_smooth = None
        # optional runtime mass clip (raw scale before log1p/z)
        try:
            self.oc_runtime_mass_clip = float(oc.get('runtime_mass_clip')) if isinstance(oc, dict) and oc.get('runtime_mass_clip') is not None else None
        except Exception:
            self.oc_runtime_mass_clip = None

        # gating MLP (always instantiate for checkpoint compatibility)
        gate_hidden = max(8, hidden_dim // 4)
        self.vel_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1)
        )
        self.mass_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1)
        )
        # bias init to encourage moderately open gates at start (~sigmoid(0.5)=0.62)
        try:
            nn.init.constant_(self.vel_gate_mlp[-1].bias, 0.5)
        except Exception:
            pass
        try:
            nn.init.constant_(self.mass_gate_mlp[-1].bias, 0.5)
        except Exception:
            pass

        # Build scalar encoders from configuration.
        fourier_cfg = oc.get("fourier", {}) if isinstance(oc, dict) else {}
        f_enable_mass = bool(fourier_cfg.get("enable_mass", True))
        f_enable_vel = bool(fourier_cfg.get("enable_velocity", True))
        f_mode = str(fourier_cfg.get("mode", "2K+2"))
        f_K = int(fourier_cfg.get("K", 16))
        f_freq_max = float(fourier_cfg.get("freq_max", 128.0))  # shared for mass & vel
        f_dc_free = bool(fourier_cfg.get("dc_free_cos", True))

        def _make_scalar_encoder(enable: bool) -> nn.Module:
            if enable:
                return ScalarFourierMLP(hidden_dim=self.hidden_dim,
                                        K=f_K, freq_max=f_freq_max,
                                        mode=f_mode, dc_free_cos=f_dc_free)
            # fallback: keep legacy linear 1->Dh
            return nn.Linear(1, self.hidden_dim)

        self.mass_embed = _make_scalar_encoder(f_enable_mass)
        self.velocity_embed = _make_scalar_encoder(f_enable_vel)
        # --------------------------------------------------------------------------------

    @staticmethod
    def _is_main_process() -> bool:
        if not dist.is_available() or not dist.is_initialized():
            # if LOCAL_RANK exists and is not 0, treat as non-main
            lr = os.environ.get('LOCAL_RANK', None)
            return lr is None or lr == '0'
        try:
            return dist.get_rank() == 0
        except Exception:
            return True

    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)

        self.register_buffer('latent_rot', latent_rot, persistent=False)
        self.register_buffer('clip_rot', clip_rot, persistent=False)


    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self,
                              clip_f: torch.Tensor,
                              sync_f: torch.Tensor,
                              text_f: torch.Tensor,
                              *,
                              cond: Optional[dict] = None,
                              object_null_mask: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        """
        Preprocess clip/sync/text features together with optional object conditions.
        `object_null_mask` disables the object branch for samples where it is True.
        """
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        pc = PreprocessedConditions(
            clip_f=clip_f,
            sync_f=sync_f,
            text_f=text_f,
            clip_f_c=clip_f_c,
            text_f_c=text_f_c,
        )

        # Merge object conditions if they are present.
        if cond is None:
            return pc

        # frame-wise mass / vel
        frame_mass, frame_vel = self._build_framewise_object_features(cond)
        obj_global_c = self._encode_object_conditions(cond) if self.oc_use_global else None

        # Apply object_null_mask in a graph-friendly vectorized form.
        if object_null_mask is not None and object_null_mask.any():
            m = object_null_mask.view(bs, 1, 1)  # (B,1,1)
            if frame_mass is not None:
                token_mass = self.empty_obj_frame_mass.expand(bs, frame_mass.size(1), -1)
                frame_mass = torch.where(m, token_mass, frame_mass)
            if frame_vel is not None:
                token_vel = self.empty_obj_frame_vel.expand(bs, frame_vel.size(1), -1)
                frame_vel = torch.where(m, token_vel, frame_vel)
            if obj_global_c is not None:
                mg = object_null_mask.view(bs, 1)  # (B,1)
                token_global = self.empty_obj_global.expand(bs, -1)
                obj_global_c = torch.where(mg, token_global, obj_global_c)

        pc.obj_frame_mass = frame_mass
        pc.obj_frame_vel = frame_vel
        pc.obj_global_c = obj_global_c
        return pc

    def get_delta_gate_stats(self) -> dict[str, float]:
        """
        Returns a flat dict of alpha_m/alpha_v values per block for logging.
        Keys:
          - delta/alpha_m/joint_{i}, delta/alpha_v/joint_{i}
          - delta/alpha_m/fused_{i}, delta/alpha_v/fused_{i}
        """
        stats: dict[str, float] = {}

        if hasattr(self, "joint_blocks"):
            for i, blk in enumerate(self.joint_blocks):
                lb = getattr(blk, "latent_block", None)
                if lb is not None and hasattr(lb, "alpha_m") and lb.alpha_m is not None:
                    try:
                        stats[f"delta/alpha_m/joint_{i}"] = float(lb.alpha_m.detach().cpu().item())
                        stats[f"delta/alpha_v/joint_{i}"] = float(lb.alpha_v.detach().cpu().item())
                    except Exception:
                        pass

        if hasattr(self, "fused_blocks"):
            for i, fb in enumerate(self.fused_blocks):
                if hasattr(fb, "alpha_m") and fb.alpha_m is not None:
                    try:
                        stats[f"delta/alpha_m/fused_{i}"] = float(fb.alpha_m.detach().cpu().item())
                        stats[f"delta/alpha_v/fused_{i}"] = float(fb.alpha_v.detach().cpu().item())
                    except Exception:
                        pass

        # also add simple aggregates for quick dashboards
        if stats:
            m_vals = [v for k, v in stats.items() if "/alpha_m/" in k]
            v_vals = [v for k, v in stats.items() if "/alpha_v/" in k]
            if m_vals:
                stats["delta/alpha_m/mean"] = float(sum(m_vals) / len(m_vals))
                stats["delta/alpha_m/max"] = float(max(m_vals))
            if v_vals:
                stats["delta/alpha_v/mean"] = float(sum(v_vals) / len(v_vals))
                stats["delta/alpha_v/max"] = float(max(v_vals))

        # Log the interaction gain.
        try:
            stats["delta/p_gain"] = float(self.p_gain.detach().cpu().item())
        except Exception:
            pass

        return stats


    def predict_flow(self,
                     latent: torch.Tensor,
                     t: torch.Tensor,
                     conditions: PreprocessedConditions) -> torch.Tensor:
        """
        Object conditions are stored in `conditions.obj_frame_mass`,
        `conditions.obj_frame_vel`, and `conditions.obj_global_c`.
        """
        assert latent.shape[1] == self._latent_seq_len

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c

        latent = self.audio_input_proj(latent)
        global_c = self.global_cond_mlp(clip_f_c + text_f_c)  # (B,Dh)

        if conditions.obj_global_c is not None:
            global_c = global_c + conditions.obj_global_c

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B,1,Dh)

        # Prepare base and optional Δ-residuals
        extended_c_base = global_c + sync_f  # (B,N_latent,Dh)
        fam_delta = None
        fvl_delta = None

        if self.use_delta_mod:
            # Δ-mode: pass fam/fvl through residual branches; do not add to base
            fam = conditions.obj_frame_mass
            fvl = conditions.obj_frame_vel
            if fam is not None and fam.shape[1] != latent.shape[1]:
                famt = fam.transpose(1, 2)
                famt = F.interpolate(famt, size=latent.shape[1], mode='nearest-exact')
                fam = famt.transpose(1, 2)
            if fvl is not None and fvl.shape[1] != latent.shape[1]:
                fvlt = fvl.transpose(1, 2)
                fvlt = F.interpolate(fvlt, size=latent.shape[1], mode='nearest-exact')
                fvl = fvlt.transpose(1, 2)
            if self.multiply_mass_vel and (fam is not None) and (fvl is not None):
                # interaction-only in Δ: cond_m <- p, cond_v <- None
                p = self.p_norm(fam * fvl)
                fam_delta = self.p_gain * p
                fvl_delta = None
            else:
                # Keep the legacy delta path when interaction-only mode is disabled.
                fam_delta = fam
                fvl_delta = fvl

        else:
            # Legacy: directly add fam/fvl into extended_c_base
            fam = conditions.obj_frame_mass
            if fam is not None:
                if fam.shape[1] != latent.shape[1]:
                    famt = fam.transpose(1, 2)
                    famt = F.interpolate(famt, size=latent.shape[1], mode='nearest-exact')
                    fam = famt.transpose(1, 2)
            fvl = conditions.obj_frame_vel
            if fvl is not None:
                if fvl.shape[1] != latent.shape[1]:
                    fvlt = fvl.transpose(1, 2)
                    fvlt = F.interpolate(fvlt, size=latent.shape[1], mode='nearest-exact')
                    fvl = fvlt.transpose(1, 2)

            if self.multiply_mass_vel and (fam is not None) and (fvl is not None):
                # interaction-only: p = m ⊙ v
                p = self.p_norm(fam * fvl)
                extended_c_base = extended_c_base + self.p_gain * p
            else:
                # fallback: add available branches individually
                if fam is not None:
                    extended_c_base = extended_c_base + fam
                if fvl is not None:
                    extended_c_base = extended_c_base + fvl


        for block in self.joint_blocks:
            latent, clip_f, text_f = block(latent, clip_f, text_f,
                                           global_c, extended_c_base,
                                           self.latent_rot, self.clip_rot,
                                           extended_c_mass=fam_delta,
                                           extended_c_vel=fvl_delta)
        for block in self.fused_blocks:
            latent = block(latent, extended_c_base, self.latent_rot,
                           cond_m=fam_delta, cond_v=fvl_delta)

        flow = self.final_layer(latent, global_c)
        return flow

        
    def forward(self,
                latent: torch.Tensor,
                clip_f: torch.Tensor,
                sync_f: torch.Tensor,
                text_f: torch.Tensor,
                t: torch.Tensor,
                **cond) -> torch.Tensor:
        """
        `cond` may contain object_null_mask, velocity_f, mass_f, presence_mask, and related fields.
        """
        object_null_mask = cond.get('object_null_mask', None)
        conditions = self.preprocess_conditions(clip_f, sync_f, text_f,
                                                cond=cond,
                                                object_null_mask=object_null_mask)
        return self.predict_flow(latent, t, conditions)

    def _encode_object_conditions(self, cond: dict) -> Optional[torch.Tensor]:
        """
        cond may contain:
          - clip_patches: (B, T, Gh, Gw, Dp)
          - obj_patch_attn: (B, O, T, P=Gh*Gw)
          - presence_mask: (B, O, T) bool
          - mass_f: (B, O, 1) float32
          - velocity_f: (B, O, T, 1) [optional, unused in aggregation]
        Returns:
          obj_global_c: (B, hidden_dim) or None
        """
        clip_patches = cond.get('clip_patches')
        obj_patch_attn = cond.get('obj_patch_attn')
        presence = cond.get('presence_mask')
        mass_f = cond.get('mass_f')
        if clip_patches is None or obj_patch_attn is None:
            return None
        if clip_patches.shape[-1] != self.obj_cond_in.in_features:
            raise ValueError(
                f"Expected {self.obj_cond_in.in_features}-dim CLIP patch tokens, "
                f"got {clip_patches.shape[-1]}; extract patches with vit_b_16 at 224px."
            )
        try:
            B, T, Gh, Gw, Dp = clip_patches.shape
            P = Gh * Gw
            patches = clip_patches.view(B, T, P, Dp)
            attn = obj_patch_attn.to(patches.dtype)  # (B, O, T, P)

            f_obj = torch.einsum('botp,btpd->botd', attn, patches)

            if presence is not None:
                pres = presence.to(f_obj.dtype).unsqueeze(-1)  # (B, O, T, 1)
                f_obj = f_obj * pres

            if mass_f is not None:
                w_mass = mass_f[..., 0]  # (B, O)
                w_mass = w_mass.clamp_min(0).unsqueeze(-1)  # (B, O, 1)
            else:
                w_mass = None

            if presence is not None:
                w_time = presence.to(f_obj.dtype)  # (B, O, T)
            else:
                w_time = (f_obj.abs().sum(dim=-1) > 0).to(f_obj.dtype)  # (B, O, T)

            w = w_time
            if w_mass is not None:
                w = w * w_mass  # (B, O, T)

            denom = w.sum(dim=(1, 2)).clamp_min(1e-5).unsqueeze(-1)  # (B,1)
            num = (f_obj * w.unsqueeze(-1)).sum(dim=(1, 2))  # (B, Dp)
            obj_feat = num / denom  # (B, Dp)

            obj_c = self.obj_cond_in(obj_feat)  # (B, hidden_dim)
            return obj_c
        except Exception as exc:
            _warn_object_condition_failure('global_object_feature_failed', exc)
            return None

    def _build_framewise_object_features(self, cond: dict) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build a frame-aligned `(B, T, D_hidden)` feature and add it to `clip_f`.
        Expected `cond` inputs:
          - clip_patches: (B, T, Gh, Gw, Dp)
          - obj_patch_attn: (B, O, T, P=Gh*Gw)
          - presence_mask: (B, O, T) bool
          - mass_f: (B, O, 1)
          - velocity_f: (B, O, T, 1)
        """
        clip_patches = cond.get('clip_patches')
        obj_patch_attn = cond.get('obj_patch_attn')
        presence = cond.get('presence_mask')
        mass_f = cond.get('mass_f')
        velocity_f = cond.get('velocity_f')
        empty_mask = cond.get('obj_empty_mask')  # (B,) bool optional
        vel_missing = cond.get('velocity_missing')  # (B,O,T) bool optional (frames where object absent)

        if clip_patches is None:
            return None, None
        if clip_patches.shape[-1] != self.patch_to_hidden.in_features:
            raise ValueError(
                f"Expected {self.patch_to_hidden.in_features}-dim CLIP patch tokens, "
                f"got {clip_patches.shape[-1]}; extract patches with vit_b_16 at 224px."
            )

        try:
            B, T, Gh, Gw, Dp = clip_patches.shape
            P = Gh * Gw

            patches = clip_patches.view(B, T, P, Dp)
            attn = obj_patch_attn.to(patches.dtype)  # (B, O, T, P) -> binary mask

            # Standard weighted sum over patches -> (B, O, T, Dp)
            f_obj = torch.einsum('botp,btpd->botd', attn, patches)

            # project patches to hidden
            f_obj_h = self.patch_to_hidden(f_obj)  # (B, O, T, Dh) -> Dh = 448



            # inject object occlusion token for absent frames (presence=False)
            if presence is not None:
                try:
                    occ_mask = ~presence  # (B,O,T) bool
                    if occ_mask.any():
                        tok_h = self.obj_occ_token_h.view(1, 1, 1, -1)  # (1,1,1,Dh)
                        f_obj_h = torch.where(occ_mask.unsqueeze(-1), tok_h, f_obj_h)
                except Exception:
                    pass



            # pre LayerNorm before FiLM branches
            if self.oc_obj_pre_ln is not None:
                try:
                    f_obj_h = self.oc_obj_pre_ln(f_obj_h)
                except Exception:
                    pass

            # presence as weights
            if presence is not None:
                w_pres = presence.to(f_obj_h.dtype)  # (B, O, T)
            else:
                w_pres = (f_obj_h.abs().sum(dim=-1) > 0).to(f_obj_h.dtype)

            # ----------------------- MASS branch (Fourier-aware) -----------------------
            if mass_f is not None:
                m_in = mass_f
                m_in = m_in.clamp_max(self.oc_runtime_mass_clip) if self.oc_runtime_mass_clip is not None else m_in

                if self.oc_mass_use_log1p:
                    try:
                        m_in_t = torch.log1p(m_in.clamp_min(0.0))
                    except Exception:
                        m_in_t = m_in.clamp_min(0.0)
                else:
                    m_in_t = m_in.clamp_min(0.0)

                if self.oc_mass_use_z:
                    try:
                        if self.oc_mass_use_log1p:
                            mean = self.oc_mass_log_mean.view(1,1,1)
                            std = self.oc_mass_log_std.view(1,1,1).clamp_min(self.oc_norm_eps)
                        else:
                            mean = self.oc_mass_mean.view(1,1,1)
                            std = self.oc_mass_std.view(1,1,1).clamp_min(self.oc_norm_eps)
                        m_in_t = (m_in_t - mean) / std
                        if self.oc_norm_clip is not None:
                            clipv = float(self.oc_norm_clip)
                            m_in_t = m_in_t.clamp(-clipv, clipv)
                    except Exception:
                        pass

                # Zero the branch before the Fourier features when the object never appears.
                if presence is not None:
                    pres_any = presence.any(dim=2, keepdim=True).to(m_in_t.dtype)  # (B,O,1)
                    m_in_star = m_in_t * pres_any
                else:
                    m_in_star = m_in_t
                # ----------------------------------------------------------------------------

                # NOTE: self.mass_embed may be Fourier encoder or Linear fallback
                m_emb = self.mass_embed(m_in_star)  # (B, O, Dh)
                m_aff = self.mass_to_affine(m_emb)  # (B, O, 2Dh)
                m_shift, m_scale = m_aff.chunk(2, dim=-1)  # (B,O,Dh)

                m_scale = 0.5 * torch.tanh(m_scale)
                m_shift = 0.5 * torch.tanh(m_shift)

                m_shift = m_shift.unsqueeze(2).expand(-1, -1, T, -1)
                m_scale = m_scale.unsqueeze(2).expand(-1, -1, T, -1)
                f_mass_obj = (f_obj_h * (1 + m_scale)) + m_shift  # (B,O,T,Dh)\

                if self.oc_ts_enable and self.mass_temporal_smooth is not None:
                    try:
                        B_, O_, T_, Dh_ = f_mass_obj.shape
                        f_mass_obj_rs = f_mass_obj.view(B_ * O_, T_, Dh_).transpose(1, 2)  # (B*O, Dh, T)
                        f_mass_obj_rs = f_mass_obj_rs + self.mass_temporal_smooth(f_mass_obj_rs)
                        f_mass_obj = f_mass_obj_rs.transpose(1, 2).view(B_, O_, T_, Dh_)
                    except Exception:
                        pass
                        
                if self.oc_mass_pool_mode == 'gate':
                    g_m = self.mass_gate_mlp(f_mass_obj)  # (B,O,T,1)
                    g_m = torch.sigmoid(g_m)
                    try:
                        g_m = g_m * (0.5 + 0.5 * w_pres.unsqueeze(-1))
                    except Exception:
                        pass
                    denom_m = g_m.sum(dim=1).clamp_min(1e-5)  # (B,T,1)
                    f_mass_t = (f_mass_obj * g_m).sum(dim=1) / denom_m  # (B,T,Dh)
                    if self.arch_debug and self._is_main_process():
                        try:
                            print('[OC][gate] mass gates stats: mean={:.4f} min={:.4f} max={:.4f}'.format(
                                g_m.mean().item(), g_m.min().item(), g_m.max().item()), flush=True)
                        except Exception:
                            pass
                else:
                    w_mass = w_pres  # presence-weighted mean
                    denom_m = w_mass.sum(dim=1).clamp_min(1e-5).unsqueeze(-1)  # (B, T, 1)
                    f_mass_t = (f_mass_obj * w_mass.unsqueeze(-1)).sum(dim=1) / denom_m  # (B, T, Dh)
                if self.arch_debug and self._is_main_process():
                    try:
                        print('[OC] f_mass_t (B,T,Dh)=', tuple(f_mass_t.shape), flush=True)
                    except Exception:
                        pass
            else:
                f_mass_t = None

            # ---------------------- VELOCITY branch (Fourier-aware) ----------------------
            if velocity_f is not None:
                v_in = velocity_f[..., 0:1]
                if self.oc_vel_use_log1p:
                    try:
                        v_in_t = torch.log1p(v_in.clamp_min(0.0))
                    except Exception:
                        v_in_t = v_in
                else:
                    v_in_t = v_in

                if self.oc_norm_enable and self.oc_vel_use_z:
                    try:
                        if self.oc_vel_use_log1p:
                            mean = self.oc_vel_log_mean.view(1,1,1,1)
                            std = self.oc_vel_log_std.view(1,1,1,1).clamp_min(self.oc_norm_eps)
                        else:
                            mean = self.oc_vel_mean.view(1,1,1,1)
                            std = self.oc_vel_std.view(1,1,1,1).clamp_min(self.oc_norm_eps)
                        v_in_t = (v_in_t - mean) / std
                        if self.oc_norm_clip is not None:
                            clipv = float(self.oc_norm_clip)
                            v_in_t = v_in_t.clamp(-clipv, clipv)
                    except Exception:
                        pass

                # Zero the branch before the Fourier features on absent frames.
                if presence is not None:
                    v_valid = presence.to(v_in_t.dtype).unsqueeze(-1)  # (B,O,T,1)
                    v_in_star = v_in_t * v_valid
                else:
                    v_in_star = v_in_t
                # -------------------------------------------------------

                # Fourier or Linear fallback
                v_emb = self.velocity_embed(v_in_star)  # (B,O,T,1)->(B,O,T,Dh)
                
                # Replace missing frames with vel_occ_token AFTER embedding, BEFORE FiLM
                if vel_missing is not None:
                    try:
                        mask_vm = vel_missing
                        if mask_vm.any():
                            occ_tok = self.vel_occ_token.view(1, 1, 1, -1)  # (1,1,1,Dh)
                            v_emb = torch.where(mask_vm.unsqueeze(-1), occ_tok, v_emb)
                    except Exception:
                        pass
                        
                v_aff = self.vel_to_affine(v_emb)  # (B,O,T,2Dh)
                v_shift, v_scale = v_aff.chunk(2, dim=-1)

                v_scale = 0.5 * torch.tanh(v_scale)
                v_shift = 0.5 * torch.tanh(v_shift)

                f_vel_obj = (f_obj_h * (1 + v_scale)) + v_shift

                if self.oc_ts_enable and self.vel_temporal_smooth is not None:
                    try:
                        B_, O_, T_, Dh_ = f_vel_obj.shape
                        f_vel_obj_rs = f_vel_obj.view(B_ * O_, T_, Dh_).transpose(1, 2)  # (B*O, Dh, T)
                        f_vel_obj_rs = f_vel_obj_rs + self.vel_temporal_smooth(f_vel_obj_rs)
                        f_vel_obj = f_vel_obj_rs.transpose(1, 2).view(B_, O_, T_, Dh_)
                    except Exception:
                        pass

                if self.oc_vel_pool_mode == 'gate':
                    g = self.vel_gate_mlp(f_vel_obj)  # (B,O,T,1)
                    g = torch.sigmoid(g)
                    try:
                        g = g * (0.5 + 0.5 * w_pres.unsqueeze(-1))
                    except Exception:
                        pass
                    denom_v = g.sum(dim=1).clamp_min(1e-5)
                    f_vel_t = (f_vel_obj * g).sum(dim=1) / denom_v  # (B,T,Dh)

                else:
                    w_vel = w_pres
                    denom_v = w_vel.sum(dim=1).clamp_min(1e-5).unsqueeze(-1)
                    f_vel_t = (f_vel_obj * w_vel.unsqueeze(-1)).sum(dim=1) / denom_v  # (B, T, Dh)

            else:
                f_vel_t = None

            # normalize each branch separately (if exists) - post pooling
            f_mass_t = self.mass_fuse_norm(f_mass_t) if isinstance(f_mass_t, torch.Tensor) else None
            f_vel_t = self.vel_fuse_norm(f_vel_t) if isinstance(f_vel_t, torch.Tensor) else None

            # apply lambda scales from config
            if isinstance(f_mass_t, torch.Tensor):
                f_mass_t = f_mass_t * self.oc_lambda_mass
            if isinstance(f_vel_t, torch.Tensor):
                f_vel_t = f_vel_t * self.oc_lambda_vel
            return f_mass_t, f_vel_t

        except Exception as exc:
            if self.arch_debug:
                raise RuntimeError("Failed to build framewise object features") from exc
            _warn_object_condition_failure('framewise_object_feature_failed', exc)
            return None, None

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_conditions(self,
                             bs: int,
                             *,
                             negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        """
        Build the unconditional CFG path with empty clip/sync/text features and zero object tensors.
        A learnable empty object token could be introduced here if needed.
        """
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)
        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)

        base = self.preprocess_conditions(empty_clip, empty_sync, empty_text)
        # Expand to the requested batch size.
        base.clip_f = base.clip_f.expand(bs, -1, -1)
        base.sync_f = base.sync_f.expand(bs, -1, -1)
        base.clip_f_c = base.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            base.text_f = base.text_f.expand(bs, -1, -1)
            base.text_f_c = base.text_f_c.expand(bs, -1)

        base.obj_frame_mass = self.empty_obj_frame_mass.expand(
            bs, self._latent_seq_len, -1
        )
        base.obj_frame_vel = self.empty_obj_frame_vel.expand(
            bs, self._latent_seq_len, -1
        )
        base.obj_global_c = self.empty_obj_global.expand(bs, -1)
        return base

    def ode_wrapper(self,
                    t: torch.Tensor,
                    latent: torch.Tensor,
                    conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions,
                    cfg_strength: float) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        # Difference between the object-conditioned path and the empty path.
        return (cfg_strength * self.predict_flow(latent, t, conditions) +
                (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))



    def load_weights(self, src_dict) -> None:
        # Accept either a plain state_dict or a checkpoint wrapper with 'weights'
        state = src_dict.get('weights', src_dict) if isinstance(src_dict, dict) else src_dict
        # Drop known ephemeral buffers if present
        for k in ['t_embed.freqs', 'latent_rot', 'clip_rot']:
            if isinstance(state, dict) and k in state:
                del state[k]

        if isinstance(state, dict):
            model_sd = self.state_dict()
            filtered = {}
            for k, v in state.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    filtered[k] = v
            missing, unexpected = self.load_state_dict(filtered, strict=False)
            if logging.getLogger().isEnabledFor(logging.INFO) and self._is_main_process():
                try:
                    print(f"[load_weights] Loaded {len(filtered)} params; missing={len(missing)} unexpected={len(unexpected)}",
                          flush=True)
                    if len(missing) > 0:
                        print(f"[load_weights] missing keys (head): {missing[:200]}", flush=True)
                    if len(unexpected) > 0:
                        print(f"[load_weights] unexpected keys (head): {unexpected[:200]}", flush=True)
                except Exception:
                    pass
        else:
            self.load_state_dict(state, strict=False)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len


def small_16k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def small_44k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   v2=True,
                   **kwargs)


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == 'small_16k':
        return small_16k(**kwargs)
    if name == 'small_44k':
        return small_44k(**kwargs)
    if name == 'medium_44k':
        return medium_44k(**kwargs)
    if name == 'large_44k':
        return large_44k(**kwargs)
    if name == 'large_44k_v2':
        return large_44k_v2(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_mmaudio('small_16k')

    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
