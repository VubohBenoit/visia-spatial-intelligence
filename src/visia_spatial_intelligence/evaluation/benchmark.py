"""Runtime benchmarking with cold-start and steady-state timings reported separately.

The benchmark records the hardware and package versions alongside the numbers, so a
result can be reproduced or rejected on another machine. Only measured values are
written; nothing is estimated.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from visia_spatial_intelligence.detection.detector import DetrDetector, ModelUnavailableError
from visia_spatial_intelligence.pipeline import VisiaPipeline

STAGES = ("detection_ms", "depth_ms", "tracking_ms", "reasoning_ms", "total_ms")


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregated timings: cold start (model loading) and steady state (after warm-up)."""

    cold_start_ms: dict[str, float | str]
    steady_state_ms: dict[str, dict[str, float]]
    fps_median: float
    samples: int
    warmup_runs: int
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_start_ms": self.cold_start_ms,
            "steady_state_ms": self.steady_state_ms,
            "fps_from_median_total": round(self.fps_median, 3),
            "samples": self.samples,
            "warmup_runs": self.warmup_runs,
            "images": self.images,
        }


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))
    return {
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "min": round(ordered[0], 3),
        "p90": round(ordered[p90_index], 3),
        "max": round(ordered[-1], 3),
    }


def benchmark_pipeline(
    pipeline: VisiaPipeline,
    images: list[tuple[str, Image.Image]],
    runs: int = 5,
    warmup_runs: int = 1,
) -> BenchmarkReport:
    """Measure load time, warm-up, then ``runs`` passes over every image.

    Raises:
        ValueError: if no images or a non-positive number of runs is given.
        ModelUnavailableError: if the required detector cannot be loaded.
    """
    if not images:
        raise ValueError("At least one image is required")
    if runs < 1:
        raise ValueError("runs must be >= 1")

    cold: dict[str, float | str] = {}
    if isinstance(pipeline.detector, DetrDetector):
        start = time.perf_counter()
        pipeline.detector.load()
        cold["detector_load_ms"] = _elapsed_ms(start)
    if pipeline.depth_estimator is not None:
        start = time.perf_counter()
        try:
            pipeline.depth_estimator.load()
            cold["depth_load_ms"] = _elapsed_ms(start)
        except ModelUnavailableError as exc:
            cold["depth_load_error"] = str(exc)

    pipeline.reset()
    for _ in range(warmup_runs):
        for _, image in images:
            pipeline.analyze(image, frame_index=0)
    pipeline.reset()

    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    per_image: list[dict[str, Any]] = []
    for name, image in images:
        image_totals: list[float] = []
        for run in range(runs):
            result = pipeline.analyze(image, frame_index=run).result
            for stage in STAGES:
                samples[stage].append(getattr(result.timings, stage))
            image_totals.append(result.timings.total_ms)
        per_image.append(
            {
                "image": name,
                "width": image.width,
                "height": image.height,
                "total_ms": _summary(image_totals),
            }
        )

    steady = {stage: _summary(values) for stage, values in samples.items()}
    median_total = steady["total_ms"]["median"]
    fps = 1000.0 / median_total if median_total > 0 else 0.0
    return BenchmarkReport(
        cold_start_ms=cold,
        steady_state_ms=steady,
        fps_median=fps,
        samples=len(samples["total_ms"]),
        warmup_runs=warmup_runs,
        images=per_image,
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def write_report(report: BenchmarkReport, path: Path, metadata: dict[str, Any]) -> None:
    """Write the report and the environment metadata as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "benchmark": report.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
