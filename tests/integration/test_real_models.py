"""Integration tests that run the real DETR and Depth Anything V2 checkpoints.

These download roughly 270 MB of weights on first use and are excluded from the default
test run. Execute them with ``make test-integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visia_spatial_intelligence.config import load_config
from visia_spatial_intelligence.pipeline import VisiaPipeline
from visia_spatial_intelligence.validation import load_image_path

pytestmark = pytest.mark.integration

SAMPLE = (
    Path(__file__).resolve().parents[2] / "data" / "samples" / "amsterdam_shopping_street_cc0.jpg"
)


@pytest.fixture(scope="module")
def sample_output():  # type: ignore[no-untyped-def]
    config = load_config()
    image = load_image_path(SAMPLE, config.limits)
    pipeline = VisiaPipeline.from_config(config)
    pipeline.reset()
    return pipeline.analyze(image, frame_index=0)


def test_detector_finds_people_in_street_scene(sample_output) -> None:  # type: ignore[no-untyped-def]
    labels = [d.label for d in sample_output.result.detections]
    assert labels.count("person") >= 3
    assert all(0.5 <= d.score <= 1.0 for d in sample_output.result.detections)


def test_detections_are_inside_image_bounds(sample_output) -> None:  # type: ignore[no-untyped-def]
    w, h = sample_output.result.width, sample_output.result.height
    for det in sample_output.result.detections:
        assert 0 <= det.box.x1 <= det.box.x2 <= w
        assert 0 <= det.box.y1 <= det.box.y2 <= h


def test_depth_is_relative_and_matches_image_size(sample_output) -> None:  # type: ignore[no-untyped-def]
    depth = sample_output.depth
    assert depth is not None
    assert depth.values.shape == (sample_output.result.height, sample_output.result.width)
    assert float(depth.values.min()) >= 0.0 and float(depth.values.max()) <= 1.0
    assert sample_output.result.disabled_features == {}
