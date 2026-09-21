# VGGSound Feature Caches

This stage prepares the MMAudio-style VGGSound feature caches that PAVAS consumes during training and cache-based evaluation.

## What This Stage Produces

PAVAS expects the standard offline features produced by the MMAudio pipeline, including:

- latent mean and standard deviation
- CLIP video features
- synchronization features
- CLIP text features

These are not produced by `train.py` on the fly. They must already exist on disk.

## How To Run This Stage

PAVAS follows the upstream MMAudio cache extraction pipeline for this step.

Reference:

- https://github.com/hkchengrex/MMAudio/blob/main/docs/TRAINING.md

In practice, the order is:

1. prepare VGGSound locally
2. follow the MMAudio feature extraction instructions
3. keep the resulting memmaps and TSV files available for the PAVAS configs

## What You Should Have Before Continuing

Before moving on to CLIP patch extraction or PPE, make sure you have:

- the VGGSound videos
- the extracted feature caches
- the corresponding TSV metadata files

PAVAS training expects these inputs to be ready before fine-tuning starts.
