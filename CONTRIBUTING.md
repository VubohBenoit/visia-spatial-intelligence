# Contributing

Thanks for looking at this research prototype. The bar is reproducibility and honesty about results.

## Setup

```bash
uv sync --frozen --all-extras
make lint
make test
```

## Rules

- **No YOLO or AGPL components.** Check the licence before adding any dependency, model or dataset, and record it in `docs/LICENSE_AUDIT.md`.
- **Pin model revisions** by commit hash in `configs/default.yaml`. Never load `main`.
- **No datasets or weights in Git.** Keep them outside the repository, or in an ignored path.
- **No fabricated results.** Any number in a document must come from a script run recorded in `docs/results/`. Mark unrun benchmarks as pending.
- **Unit tests must not download models.** Put tests that need weights under `tests/integration/` with the `integration` marker.
- **Keep the reasoning deterministic and explainable.** New rules need a unit test and a `reasons` entry.

## Before opening a pull request

```bash
make lint
make test
```
