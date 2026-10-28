.PHONY: help install lint format test test-integration info analyze-sample clean

UV ?= uv
SAMPLE ?= data/samples/amsterdam_shopping_street_cc0.jpg
OUT ?= artifacts/sample

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv and install locked dependencies (CPU PyTorch)
	$(UV) sync --frozen

lint: ## Ruff lint and format check (no changes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply Ruff formatting and auto-fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test: ## Unit tests (no model downloads)
	$(UV) run pytest

test-integration: ## Tests that download and run real model weights (~270 MB on first run)
	$(UV) run pytest -m integration -v

info: ## Print runtime and pipeline configuration
	$(UV) run visia info

analyze-sample: ## Run the pipeline on the bundled CC0 sample image
	$(UV) run visia analyze $(SAMPLE) --out-dir $(OUT) --show-depth

clean: ## Remove caches and generated artifacts (keeps artifacts/.gitkeep)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	find artifacts -mindepth 1 ! -name .gitkeep -exec rm -rf {} + 2>/dev/null || true
