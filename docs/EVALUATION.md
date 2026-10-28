# Evaluation

## What exists today

| Item | Status | Where |
|---|---|---|
| Runtime benchmark (cold start, steady state, FPS, load and hardware recorded) | **Implemented and run** on one image | `visia benchmark`, `src/visia_spatial_intelligence/evaluation/benchmark.py` |
| Real-model integration checks (people detected, bounds valid, depth shape and range) | **Implemented and passing** | `tests/integration/test_real_models.py` |
| Detection precision, recall, F1, mAP@50, mAP@50:95 | **Pending.** No labelled evaluation split exists yet. | — |
| Alert precision and recall, missed-obstacle rate, false alerts per minute | **Pending.** Requires a defined ground-truth protocol and labelled video. | — |
| Ablation (A: detection, B: + tracking, C: + reasoning, D: + depth) | **Pending** | — |
| Generalisation by scene condition | **Pending** | — |

No accuracy metric has been computed. No result in this repository should be read as a
detection or alert accuracy figure.

## Measured runtime (single image, one machine)

Reproduce with:

```bash
uv run visia benchmark data/samples/amsterdam_shopping_street_cc0.jpg --runs 3 --warmup 1 \
  --out artifacts/benchmarks/latest.json
```

Hardware: Intel Core i7-1068NG7 @ 2.30 GHz, 4 cores / 8 threads, 32 GB RAM, macOS 26.6.2,
x86_64. No GPU; PyTorch MPS is unavailable on Intel Macs. PyTorch 2.2.2 (CPU), transformers
4.49.0. Input: 1280×960 JPEG.

Measured on 2026-10-28, the machine was **heavily loaded by other applications** (1-minute
load average up to about 175 on 8 logical CPUs, from a virtual machine, Safari and VS Code).
Timings from that period are therefore pessimistic and vary between runs by about 2×. They
are recorded for transparency and should be re-measured on an idle machine before any
comparison. See the JSON files under `docs/results/` for the exact numbers, including the
load average captured with each run.

Detection includes the one-time model load in the first call, which is why cold start is
reported separately. Preprocessing is inside the detector call and is not reported
separately yet.

## Reproducibility fields

Each benchmark JSON records: the command, package versions, platform, CPU count, load
average, pipeline configuration (model IDs and pinned commit hashes, thresholds, corridor),
and the seed. Weights are pinned by commit in `configs/default.yaml`, and SHA-256 digests are
listed in `LICENSE_AUDIT.md`.
