from typing import Literal, Optional
import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
try:
    import timm  # optional, for DINOv2 via timm
except Exception:
    timm = None

from core.ext.autoencoder import AutoEncoderModule
from core.ext.mel_converter import get_mel_converter
from core.ext.synchformer import Synchformer
from core.model.utils.distributions import DiagonalGaussianDistribution


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
        clip_model: str = 'vit_b_16',
    ):
        super().__init__()

        if enable_conditions:
            # Select vision backbone: CLIP or DINOv2
            self.vision_backend = 'clip'
            self.clip_model = None
            self.dino_model = None
            tokenizer_id = None
            if clip_model in ('vit_b_16', 'vit_l_14', 'vit_h_14_384'):
                self.vision_backend = 'clip'
                model_id = None
                tokenizer_id = None
                if clip_model == 'vit_b_16':
                    model_id = 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K'
                    tokenizer_id = 'ViT-B-16'
                elif clip_model == 'vit_l_14':
                    model_id = 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
                    tokenizer_id = 'ViT-L-14'
                elif clip_model == 'vit_h_14_384':
                    model_id = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
                    tokenizer_id = 'ViT-H-14-378-quickgelu'
                self.clip_model = create_model_from_pretrained(model_id, return_transform=False)
                self.clip_preprocess = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                 std=[0.26862954, 0.26130258, 0.27577711])
                self.clip_model = patch_clip(self.clip_model)
            elif clip_model == 'dinov2_s_14':
                self.vision_backend = 'dino'
                # Try to load DINOv2-S/14 model
                loaded = False
                self.dino_preprocess = Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                if timm is not None:
                    for name in (
                        'vit_small_patch14_dinov2.lvd142m',
                        'vit_small_patch14_dinov2',
                    ):
                        try:
                            self.dino_model = timm.create_model(name, pretrained=True, num_classes=0)
                            loaded = True
                            break
                        except Exception:
                            continue
                if not loaded:
                    try:
                        # Fallback to torch.hub
                        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                        loaded = True
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load DINOv2-S/14 via timm or torch.hub: {e}. Please install timm or allow hub access.")
            else:
                raise ValueError(f'Unknown clip_model: {clip_model}')

            # Load Synchformer only if a checkpoint is provided
            if synchformer_ckpt is not None:
                self.synchformer = Synchformer()
                self.synchformer.load_state_dict(
                    torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))
            else:
                self.synchformer = None

            self.tokenizer = open_clip.get_tokenizer(tokenizer_id) if tokenizer_id is not None else None
        else:
            self.clip_model = None
            self.dino_model = None
            self.synchformer = None
            self.tokenizer = None

        if tod_vae_ckpt is not None:
            self.mel_converter = get_mel_converter(mode)
            self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                         vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                         mode=mode,
                                         need_vae_encoder=need_vae_encoder)
        else:
            self.tod = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.encode_image(x[i:i + batch_size], normalize=True))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_clip_patches(
        self,
        x: torch.Tensor,
        batch_size: int = -1,
        return_grid: bool = True,
        normalize: bool = False,
        include_cls: bool = False,
    ):
        """
        Extract patch-wise tokens for each frame using the selected vision transformer (CLIP or DINOv2),
        returning per-frame patch embeddings without applying the final projection head.

        Args:
            x: (B, T, C, H, W)
            batch_size: process (B*T) frames per chunk.
            return_grid: if True returns shape (B, T, Gh, Gw, D), else (B, T, S, D).
            normalize: L2-normalize tokens across the last dim if True.
            include_cls: also return (B, T, D) CLS tokens if True.
        """
        assert self.vision_backend in ('clip', 'dino')

        # Preprocess
        b, t, c, h, w = x.shape
        assert c == 3, 'Expected RGB frames'
        if self.vision_backend == 'clip':
            assert self.clip_model is not None, 'CLIP is not loaded'
            v = self.clip_model.visual
            x = self.clip_preprocess(x)
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        else:
            assert self.dino_model is not None, 'DINO is not loaded'
            v = self.dino_model
            x = self.dino_preprocess(x)
            x = rearrange(x, 'b t c h w -> (b t) c h w')

        if batch_size < 0:
            batch_size = b * t

        def _interp_pos_clip(pos_emb: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
            """Interpolate CLIP positional embeddings to target grid size.
            pos_emb: (1, 1+S0, D) -> (1, 1+gh*gw, D)
            """
            cls_pos = pos_emb[:, :1]
            patch_pos = pos_emb[:, 1:]  # (1, S0, D)
            s0 = patch_pos.shape[1]
            target_s = gh * gw
            if s0 == target_s:
                return pos_emb
            root = int(math.isqrt(s0))
            if root * root == s0:
                patch_pos_2d = patch_pos.reshape(1, root, root, -1).permute(0, 3, 1, 2)
                patch_pos_2d = F.interpolate(patch_pos_2d, size=(gh, gw), mode='bicubic', align_corners=False)
                patch_pos = patch_pos_2d.permute(0, 2, 3, 1).reshape(1, target_s, -1)
            else:
                patch_pos_1d = patch_pos.permute(0, 2, 1)  # (1, D, S0)
                patch_pos_1d = F.interpolate(patch_pos_1d, size=target_s, mode='linear', align_corners=False)
                patch_pos = patch_pos_1d.permute(0, 2, 1)  # (1, target_s, D)
            return torch.cat([cls_pos, patch_pos], dim=1)

        def _encode_tokens_clip(img_batch: torch.Tensor):
            v_clip = self.clip_model.visual
            cast_dtype = v_clip.conv1.weight.dtype
            z = img_batch.to(dtype=cast_dtype)
            z = v_clip.conv1(z)  # (N, width, gh, gw)
            _, width, gh, gw = z.shape
            z = z.reshape(z.shape[0], width, gh * gw).permute(0, 2, 1)
            cls_tok = v_clip.class_embedding.to(z.dtype)
            cls_tok = cls_tok.unsqueeze(0).unsqueeze(1).expand(z.shape[0], -1, -1)
            z = torch.cat([cls_tok, z], dim=1)
            pos = v_clip.positional_embedding.to(z.dtype)
            if pos.ndim == 2:
                pos = pos.unsqueeze(0)
            if pos.shape[1] != z.shape[1]:
                pos = _interp_pos_clip(pos, gh, gw)
            z = z + pos
            z = v_clip.ln_pre(z)
            z = z.permute(1, 0, 2)
            z = v_clip.transformer(z)
            z = z.permute(1, 0, 2)
            z = v_clip.ln_post(z)
            cls_out = z[:, 0, :]
            patch_tokens = z[:, 1:, :]
            return patch_tokens, cls_out

        def _encode_tokens_dino(img_batch: torch.Tensor):
            v_dino = self.dino_model
            # timm ViT: patch_embed -> add cls token -> add/interp pos_embed -> blocks -> norm
            z = v_dino.patch_embed(img_batch)  # (N, S, D)
            N, S, D = z.shape
            # Determine grid by inference later
            # Add cls token if exists
            cls_out = None
            if getattr(v_dino, 'cls_token', None) is not None:
                cls_tok = v_dino.cls_token.expand(N, -1, -1).to(z.dtype)
                z = torch.cat((cls_tok, z), dim=1)  # (N, 1+S, D)
            # Positional embedding
            if getattr(v_dino, 'pos_embed', None) is not None:
                pos = v_dino.pos_embed.to(z.dtype)
                # Interpolate patch part if mismatch
                if pos.shape[1] != z.shape[1]:
                    # Assume pos has cls + patches
                    cls_pos = pos[:, :1]
                    patch_pos = pos[:, 1:]
                    s0 = patch_pos.shape[1]
                    root = int(math.isqrt(s0))
                    gh_gw = int(math.isqrt(S))
                    if root * root == s0 and gh_gw * gh_gw == S:
                        patch_pos_2d = patch_pos.reshape(1, root, root, -1).permute(0, 3, 1, 2)
                        patch_pos_2d = F.interpolate(patch_pos_2d, size=(gh_gw, gh_gw), mode='bicubic', align_corners=False)
                        patch_pos = patch_pos_2d.permute(0, 2, 3, 1).reshape(1, S, -1)
                        pos = torch.cat([cls_pos, patch_pos], dim=1)
                    else:
                        # Fallback linear along sequence
                        patch_pos_1d = patch_pos.permute(0, 2, 1)
                        patch_pos_1d = F.interpolate(patch_pos_1d, size=S, mode='linear', align_corners=False)
                        patch_pos = patch_pos_1d.permute(0, 2, 1)
                        pos = torch.cat([cls_pos, patch_pos], dim=1)
                z = z + pos
            # Some timm models have pre-norm
            if hasattr(v_dino, 'norm_pre') and v_dino.norm_pre is not None:
                z = v_dino.norm_pre(z)
            # Blocks
            for blk in v_dino.blocks:
                z = blk(z)
            # Final norm
            if hasattr(v_dino, 'norm') and v_dino.norm is not None:
                z = v_dino.norm(z)
            # Separate cls if present
            if getattr(v_dino, 'cls_token', None) is not None:
                cls_out = z[:, 0, :]
                patch_tokens = z[:, 1:, :]
            else:
                patch_tokens = z
            return patch_tokens, cls_out

        toks = []
        clses = [] if include_cls else None
        for i in range(0, b * t, batch_size):
            if self.vision_backend == 'clip':
                pb, cb = _encode_tokens_clip(x[i:i + batch_size])
            else:
                pb, cb = _encode_tokens_dino(x[i:i + batch_size])
            toks.append(pb)
            if include_cls and cb is not None:
                clses.append(cb)

        toks = torch.cat(toks, dim=0)  # ((b*t), S, D)
        if normalize:
            toks = F.normalize(toks, dim=-1)

        if include_cls and clses is not None:
            clses = torch.cat(clses, dim=0)
            if normalize:
                clses = F.normalize(clses, dim=-1)

        if return_grid:
            # Infer grid size from token length
            S = toks.shape[1]
            inferred = int(math.isqrt(S))
            if inferred * inferred == S:
                gh_final, gw_final = inferred, inferred
            else:
                # Fallback to a single-row grid if not square
                gh_final, gw_final = 1, S
            toks = rearrange(toks, '(b t) (h w) d -> b t h w d', b=b, h=gh_final, w=gw_final)
        else:
            toks = rearrange(toks, '(b t) s d -> b t s d', b=b)

        if include_cls and clses is not None:
            clses = rearrange(clses, '(b t) d -> b t d', b=b)
            return toks, clses
        return toks

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384

        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
