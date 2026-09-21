from typing import Optional

from nitrous_ema import PostHocEMA
from omegaconf import DictConfig

from core.model.networks import get_my_mmaudio


def synthesize_ema(cfg: DictConfig, sigma: float, step: Optional[int]):
    # vae = get_my_mmaudio(cfg.model, use_delta_mod=getattr(cfg, 'use_delta_mod', False))  # NEW
    vae = get_my_mmaudio(cfg.model, use_delta_mod=getattr(cfg, 'use_delta_mod', False), multiply_mass_vel=getattr(cfg, 'multiply_mass_vel', False))  # NEW
    emas = PostHocEMA(vae,
                      sigma_rels=cfg.ema.sigma_rels,
                      update_every=cfg.ema.update_every,
                      checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                      checkpoint_folder=cfg.ema.checkpoint_folder)

    try:
        synthesized_ema = emas.synthesize_ema_model(sigma_rel=sigma, step=step, device='cpu')
    except ValueError as exc:
        if 'empty sequence' in str(exc):
            return None
        raise
    state_dict = synthesized_ema.ema_model.state_dict()
    return state_dict
