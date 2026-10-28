"""Command-line interface.

Examples::

    visia info
    visia analyze path/to/image.jpg --out-dir artifacts/run1 --show-depth

Exit codes: 0 success, 2 invalid input or configuration, 3 required model unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

from visia_spatial_intelligence import __version__
from visia_spatial_intelligence.config import AppConfig, ConfigError, load_config
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.evaluation.benchmark import benchmark_pipeline, write_report
from visia_spatial_intelligence.pipeline import VisiaPipeline
from visia_spatial_intelligence.validation import InputValidationError, load_image_path
from visia_spatial_intelligence.visualization.renderer import render_frame

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_MODEL = 3

DISCLAIMER = (
    "Research prototype. Not a certified navigation, mobility or safety system. "
    "Do not rely on it to avoid hazards."
)


def runtime_info() -> dict[str, Any]:
    """Describe the interpreter, platform and key package versions for reproducibility."""
    packages = {}
    for name in ("torch", "transformers", "numpy", "pillow", "pyyaml"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    accelerators: dict[str, bool] = {"cuda": False, "mps": False}
    try:
        import torch

        accelerators["cuda"] = bool(torch.cuda.is_available())
        accelerators["mps"] = bool(torch.backends.mps.is_available())
    except ImportError:  # pragma: no cover
        pass
    return {
        "visia_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15min": [round(v, 2) for v in os.getloadavg()],
        "packages": packages,
        "accelerators": accelerators,
    }


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(Path(args.config) if args.config else None)
    if args.no_depth:
        config = config.with_overrides(depth=replace(config.depth, enabled=False))
    if args.no_tracking:
        config = config.with_overrides(tracker=replace(config.tracker, enabled=False))
    return config


def _cmd_info(args: argparse.Namespace) -> int:
    config = _build_config(args)
    payload = {
        "runtime": runtime_info(),
        "pipeline": VisiaPipeline.from_config(config).describe(),
        "disclaimer": DISCLAIMER,
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    config = _build_config(args)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)

    try:
        image = load_image_path(image_path, config.limits)
    except InputValidationError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return EXIT_INPUT

    pipeline = VisiaPipeline.from_config(config)
    pipeline.reset()
    try:
        output = pipeline.analyze(image, frame_index=0)
    except ModelUnavailableError as exc:
        print(f"Model unavailable: {exc}", file=sys.stderr)
        print(
            "Check your network connection or the model id in the configuration.", file=sys.stderr
        )
        return EXIT_MODEL

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = render_frame(
        image,
        output.result,
        config.reasoning.corridor,
        depth=output.depth,
        show_depth=args.show_depth,
    )
    annotated_path = out_dir / "annotated.png"
    annotated.save(annotated_path)

    report = {
        "disclaimer": DISCLAIMER,
        "input": {"file": image_path.name, "width": image.width, "height": image.height},
        "pipeline": pipeline.describe(),
        "runtime": runtime_info(),
        "result": output.result.to_dict(),
    }
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Detections: {len(output.result.detections)}")
    print(f"Alerts: {len(output.result.alerts)}")
    print(f"Disabled features: {output.result.disabled_features or 'none'}")
    print(f"Annotated image: {annotated_path}")
    print(f"JSON report: {json_path}")
    return EXIT_OK


def _cmd_benchmark(args: argparse.Namespace) -> int:
    config = _build_config(args)
    images = []
    for path_text in args.images:
        path = Path(path_text)
        try:
            images.append((path.name, load_image_path(path, config.limits)))
        except InputValidationError as exc:
            print(f"Invalid input {path.name}: {exc}", file=sys.stderr)
            return EXIT_INPUT

    pipeline = VisiaPipeline.from_config(config)
    try:
        report = benchmark_pipeline(pipeline, images, runs=args.runs, warmup_runs=args.warmup)
    except ModelUnavailableError as exc:
        print(f"Model unavailable: {exc}", file=sys.stderr)
        return EXIT_MODEL

    metadata = {
        "disclaimer": DISCLAIMER,
        "runtime": runtime_info(),
        "pipeline": pipeline.describe(),
        "command": "visia benchmark " + " ".join(args.images),
    }
    out_path = Path(args.out)
    write_report(report, out_path, metadata)
    print(json.dumps(report.to_dict()["fps_from_median_total"]), "FPS (median end-to-end)")
    print(f"Benchmark report: {out_path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visia", description=DISCLAIMER)
    parser.add_argument("--config", help="Path to a YAML configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    feature_flags = argparse.ArgumentParser(add_help=False)
    feature_flags.add_argument("--no-depth", action="store_true", help="Disable depth estimation")
    feature_flags.add_argument("--no-tracking", action="store_true", help="Disable tracking")

    sub.add_parser(
        "info", parents=[feature_flags], help="Print runtime and pipeline configuration as JSON"
    )

    analyze = sub.add_parser("analyze", parents=[feature_flags], help="Analyse a single image")
    analyze.add_argument("image", help="Path to a JPEG, PNG, BMP or WebP image")
    analyze.add_argument("--out-dir", default="artifacts/latest", help="Output directory")
    analyze.add_argument("--show-depth", action="store_true", help="Overlay relative depth")

    bench = sub.add_parser(
        "benchmark", parents=[feature_flags], help="Measure cold-start and steady-state latency"
    )
    bench.add_argument("images", nargs="+", help="One or more images to time")
    bench.add_argument("--runs", type=int, default=5, help="Measured passes per image")
    bench.add_argument("--warmup", type=int, default=1, help="Unmeasured warm-up passes")
    bench.add_argument(
        "--out", default="artifacts/benchmarks/latest.json", help="Where to write the JSON report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        return _cmd_analyze(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
