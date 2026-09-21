import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2

# Prefer torchvision VideoReader with PyAV backend if available, else fall back to torio
try:
    from torchvision.io import VideoReader  # type: ignore
    _HAS_TV_READER = True
except Exception:
    _HAS_TV_READER = False

try:
    from torio.io import StreamingMediaDecoder  # type: ignore
    _HAS_TORIO = True
except Exception:
    _HAS_TORIO = False

from core.utils.dist_utils import local_rank

log = logging.getLogger(__name__)

_CLIP_SIZE = 384
_CLIP_FPS = 8.0


class VGGSoundClipPatches(Dataset):
    """
    Minimal video-only dataset for patch extraction.
    - Loads only CLIP stream (8 FPS, 8 seconds by default)
    - No sync video, no audio
    - Recovers short videos by padding last frame; trims long ones
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = 'sets/vgg3-train.tsv',
        duration_sec: float = 8.0,
        clip_fps: float = _CLIP_FPS,
        clip_size: int = _CLIP_SIZE,
    ):
        self.root = Path(root)
        self.duration_sec = float(duration_sec)
        self.clip_expected_length = int(round(clip_fps * self.duration_sec))
        self.clip_fps = float(clip_fps)
        self.clip_size = int(clip_size)

        videos = sorted(os.listdir(self.root))
        videos = set([Path(v).stem for v in videos])
        self.labels = {}
        self.videos = []
        missing = []

        df_list = pd.read_csv(tsv_path, sep='\t', dtype={'id': str}).to_dict('records')
        for r in df_list:
            vid = r['id']
            if vid in videos:
                self.labels[vid] = r['label']
                self.videos.append(vid)
            else:
                missing.append(vid)

        if local_rank == 0:
            log.info(f'{len(videos)} files under {self.root}')
            log.info(f'{len(self.videos)} matched from {tsv_path}; {len(missing)} missing')
            if not _HAS_TV_READER and not _HAS_TORIO:
                log.warning('Neither torchvision VideoReader nor torio StreamingMediaDecoder available')

        self.clip_transform = v2.Compose([
            v2.Resize((self.clip_size, self.clip_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    @staticmethod
    def _pad_or_trim_frames(x: torch.Tensor, t_len: int) -> torch.Tensor:
        # x: (T, H, W, 3)
        if x is None or x.numel() == 0:
            return x
        t = x.shape[0]
        if t == t_len:
            return x
        if t > t_len:
            return x[:t_len]
        pad = t_len - t
        last = x[-1:].expand(pad, -1, -1, -1)
        return torch.cat([x, last], dim=0)

    def _read_with_tv(self, path: Path) -> Optional[torch.Tensor]:
        # Returns frames in (T, H, W, 3) uint8
        try:
            vr = VideoReader(str(path), 'video')
            # TV returns native fps, we downsample uniformly to expected length
            frames = []
            for f in vr:
                frames.append(f['data'])  # (H,W,3), uint8
            if len(frames) == 0:
                return None
            vid = torch.stack(frames, dim=0)
            # Uniform temporal sampling to expected length
            t = vid.shape[0]
            if t == self.clip_expected_length:
                return vid
            if t < self.clip_expected_length:
                return self._pad_or_trim_frames(vid, self.clip_expected_length)
            # downsample by linear index mapping
            idx = torch.linspace(0, t - 1, self.clip_expected_length).round().long()
            return vid[idx]
        except Exception as e:
            log.error(f'VideoReader failed on {path.name}: {e}')
            return None

    def _read_with_torio(self, path: Path) -> Optional[torch.Tensor]:
        try:
            reader = StreamingMediaDecoder(path)
            reader.add_basic_video_stream(
                frames_per_chunk=self.clip_expected_length,
                frame_rate=self.clip_fps,
                format='rgb24',
            )
            reader.fill_buffer()
            chunks = reader.pop_chunks()
            clip_chunk = chunks[0] if len(chunks) > 0 else None
            return clip_chunk
        except Exception as e:
            log.error(f'Torio failed on {path.name}: {e}')
            return None

    def __getitem__(self, idx: int):
        vid = self.videos[idx]
        path = self.root / f'{vid}.mp4'

        frames = None
        if _HAS_TV_READER:
            frames = self._read_with_tv(path)
        if frames is None and _HAS_TORIO:
            frames = self._read_with_torio(path)
        if frames is None or frames.numel() == 0:
            log.error(f'Failed to decode {vid}')
            return None

        frames = self._pad_or_trim_frames(frames, self.clip_expected_length)
        if frames is None or frames.shape[0] != self.clip_expected_length:
            log.error(f'Wrong length after pad for {vid}')
            return None

        clip = self.clip_transform(frames)
        return {
            'index': idx,
            'id': vid,
            'caption': self.labels[vid],
            'clip_video': clip,
        }

    def __len__(self):
        return len(self.videos)
