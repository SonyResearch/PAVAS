from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
cfg = OmegaConf.load(REPO_ROOT / "configs" / "paths.example.yaml")
assert "dataset" in cfg and "ppe" in cfg and "weights" in cfg
with initialize_config_dir(version_base="1.3.2", config_dir=str(REPO_ROOT / "configs")):
    train_cfg = compose(config_name="train/finetune", overrides=["model=small_16k"])
    eval_cfg = compose(config_name="eval/eval", overrides=["model=small_16k"])
assert train_cfg.num_iterations == 30_000
assert train_cfg.data.ExtractedVGG.object_cond.enabled
assert eval_cfg.dataset == "vggsound"
print("smoke_test_paths: ok")
