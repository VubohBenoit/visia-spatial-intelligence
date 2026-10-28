# Progress log

## Completed

- Environment inspection: macOS 26.6.2, Intel x86_64 (i7-1068NG7), 32 GB RAM, Python 3.11.15, uv 0.13.6. No MPS.
- Licence audit for DETR, Depth Anything V2 Small, transformers, timm, torch and torchvision (`docs/LICENSE_AUDIT.md`). Base and Large depth checkpoints rejected (CC-BY-NC).
- Core package: typed schemas, configuration, input validation, DETR detector, Depth Anything V2 estimator, IoU tracker, spatial reasoning with alerts and cooldown, renderer, pipeline, CLI.
- Unit tests (59 passing, no downloads); integration tests against real weights (3 passing).
- Real end-to-end run on a CC0 sample image; annotated output and JSON produced.
- Runtime benchmark command with cold-start and steady-state separation.
- Streamlit demo (default state verified; upload path not exercised by automated tests).
- FastAPI service with size, timeout and error handling (tested with fake pipelines).
- CI workflow (Ruff, pytest), Dockerfile, Makefile, docs.

## Pending

- Accuracy evaluation: no labelled dataset or split has been selected or audited.
- Alert ground-truth protocol and alert metrics.
- Ablations A–D.
- Re-measured runtime on an idle machine (the 2026-10-28 runs were under heavy external load).
- Walkable-area segmentation (no checkpoint has passed the licence audit).
- Video decoding.
- Browser test of the Streamlit upload flow.

## Blockers

- None blocking the baseline. Accuracy work is blocked on choosing a licence-verified labelled dataset.

## Next action

Re-run `uv run visia benchmark` on a quiet machine and save the JSON to `docs/results/`.
