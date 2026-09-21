import sys
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from precompute.clip_patches.extract_clip_patches_mp import setup_loader, should_init_dist

previous = {key: os.environ.get(key) for key in ('RANK', 'WORLD_SIZE', 'SLURM_PROCID')}
try:
    os.environ.pop('RANK', None)
    os.environ.pop('WORLD_SIZE', None)
    os.environ['SLURM_PROCID'] = '0'
    assert not should_init_dist(SimpleNamespace(merge_only=False))
    os.environ['RANK'] = '0'
    assert should_init_dist(SimpleNamespace(merge_only=False))
finally:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


with tempfile.TemporaryDirectory(prefix="pavas-clip-smoke-") as tmp:
    root = Path(tmp)
    video_root = root / "videos"
    video_root.mkdir()
    (video_root / "example.mp4").touch()
    tsv = root / "videos.tsv"
    tsv.write_text("id\tlabel\nexample\texample sound\n", encoding="utf-8")
    dataset, loader = setup_loader(video_root, tsv, batch_size=1, num_workers=0, clip_size=224)
    assert len(dataset) == 1
    assert isinstance(loader.sampler, SequentialSampler)

print("smoke_test_clip_loader: ok")
