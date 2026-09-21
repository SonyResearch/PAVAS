# Evaluation

PAVAS supports cache-based benchmark evaluation with `evaluate.py`.

References:

- MMAudio evaluation instructions: https://github.com/hkchengrex/MMAudio/blob/main/docs/EVAL.md
- `av-benchmark`: https://github.com/hkchengrex/av-benchmark

## Main Config

Start from:

- `configs/eval/eval.yaml`

Before running evaluation, confirm that:

1. your local config paths are correct
2. the checkpoint path exists
3. the output directory is writable

`configs/eval/eval.yaml` enables the same delta modulation as training.
Select the `model` variant matching your checkpoint and set the active
test-cache, CLIP patch, and PPE paths in `configs/data/base.yaml`.

## 1. Cache-Based Benchmark Evaluation

Use `evaluate.py` when the evaluation input is already organized through caches and benchmark metadata.

Example:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 \
  evaluate.py model=small_16k +test.weight_path=/path/to/pavas_finetune_last.pth
```

`evaluate.py` loads VGGSound feature caches, CLIP patch features and PPE
outputs, generates audio, extracts the prediction feature cache, then computes
metrics. Before running it, check:

1. the evaluation cache paths are valid
2. the checkpoint path is valid
3. the benchmark output directory is writable

For a one-video smoke test, append
`+data.ExtractedVGG_test.compute_metrics=false`. This still generates audio
and extracts the prediction feature cache, but skips dataset-level metrics,
which are not defined for a single prediction. The default is `true`.

Install `av-benchmark` with `scripts/setup_env.sh` (or the equivalent commands
in `INSTALL.md`) and download its public metric weights:

```bash
python scripts/download_eval_assets.py
```

## 2. Quantitative Metrics From Precomputed Results

Use `av-benchmark` for quantitative metrics:

1. obtain the VGGSound prediction cache tar from the
   [PAVAS precomputed results dataset](https://huggingface.co/datasets/HBoh/PAVAS-precomputed-results/tree/main)
2. extract `PAVAS_large_vggsound_cache.tar`
3. prepare the VGGSound test ground-truth cache using the MMAudio evaluation guide
4. run `av-benchmark` against the two cache directories

FoleyBench, VGGSounder, and FlatSound results are planned for a later release.

If predicted and ground-truth caches already exist, use the installed
`av-benchmark` API through the repository command below. Audio feature
extraction is not repeated.

```bash
python scripts/evaluate_precomputed.py \
  --gt-cache /path/to/vggsound_eval_cache \
  --pred-cache /path/to/PAVAS_large_vggsound_cache \
  --output-json /path/to/metrics.json
```
