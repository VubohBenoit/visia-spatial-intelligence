"""Pipeline tests using fake models, so no weights are downloaded."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visia_spatial_intelligence.config import AppConfig, DepthConfig, ReasoningConfig
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.pipeline import VisiaPipeline
from visia_spatial_intelligence.schemas import BoundingBox, DepthMap, Detection
from visia_spatial_intelligence.spatial.reasoning import SpatialReasoner
from visia_spatial_intelligence.tracking.tracker import IoUTracker


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, image: Image.Image) -> list[Detection]:
        return list(self._detections)


class UnavailableDetector:
    def detect(self, image: Image.Image) -> list[Detection]:
        raise ModelUnavailableError("weights not cached and network disabled")


class FakeDepth:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def estimate(self, image: Image.Image) -> DepthMap:
        if self._fail:
            raise RuntimeError("simulated inference failure")
        values = np.full((image.height, image.width), 0.9, dtype=np.float32)
        return DepthMap(values=values, height=image.height, width=image.width)


CENTRE_BOX = Detection(label="person", score=0.95, box=BoundingBox(290, 250, 350, 470))


def _pipeline(
    config: AppConfig,
    detector: object,
    depth: object | None,
) -> VisiaPipeline:
    return VisiaPipeline(
        config=config,
        detector=detector,  # type: ignore[arg-type]
        depth_estimator=depth,  # type: ignore[arg-type]
        tracker=IoUTracker(config.tracker),
        reasoner=SpatialReasoner(config.reasoning),
    )


def test_full_pipeline_with_fakes_produces_assessments_and_timings(
    app_config: AppConfig, blank_image: Image.Image
) -> None:
    pipeline = _pipeline(app_config, FakeDetector([CENTRE_BOX]), FakeDepth())
    output = pipeline.analyze(blank_image, frame_index=0)
    result = output.result
    assert len(result.detections) == 1
    assert result.detections[0].track_id == 1
    assert result.assessments[0].relative_proximity == pytest.approx(0.9)
    assert result.disabled_features == {}
    assert result.timings.total_ms >= result.timings.detection_ms >= 0.0
    assert output.depth is not None


def test_depth_failure_disables_only_depth(app_config: AppConfig, blank_image: Image.Image) -> None:
    pipeline = _pipeline(app_config, FakeDetector([CENTRE_BOX]), FakeDepth(fail=True))
    result = pipeline.analyze(blank_image).result
    assert "depth" in result.disabled_features
    assert "simulated inference failure" in result.disabled_features["depth"]
    assert result.assessments[0].relative_proximity is None


def test_configured_depth_disable_is_reported(
    app_config: AppConfig, blank_image: Image.Image
) -> None:
    config = app_config.with_overrides(depth=DepthConfig(enabled=False))
    pipeline = VisiaPipeline.from_config(config)
    assert pipeline.depth_estimator is None
    detector = FakeDetector([CENTRE_BOX])
    pipeline.detector = detector  # type: ignore[assignment]
    result = pipeline.analyze(blank_image).result
    assert result.disabled_features["depth"] == "disabled by configuration"


def test_missing_detector_raises_clear_error(
    app_config: AppConfig, blank_image: Image.Image
) -> None:
    pipeline = _pipeline(app_config, UnavailableDetector(), None)
    with pytest.raises(ModelUnavailableError, match="weights not cached"):
        pipeline.analyze(blank_image)


def test_no_detector_configured_raises(app_config: AppConfig, blank_image: Image.Image) -> None:
    pipeline = _pipeline(app_config, None, None)
    with pytest.raises(ModelUnavailableError, match="No object detector"):
        pipeline.analyze(blank_image)


def test_video_sequence_emits_alert_after_persistence(
    app_config: AppConfig, blank_image: Image.Image
) -> None:
    config = app_config.with_overrides(
        reasoning=ReasoningConfig(min_frames_for_alert=3, alert_cooldown_frames=50)
    )
    pipeline = _pipeline(config, FakeDetector([CENTRE_BOX]), FakeDepth())
    pipeline.reset()
    alerts = []
    for frame in range(4):
        alerts.extend(pipeline.analyze(blank_image, frame_index=frame).result.alerts)
    assert [a.frame_index for a in alerts] == [2]


def test_describe_reports_model_identifiers(app_config: AppConfig) -> None:
    info = VisiaPipeline.from_config(app_config).describe()
    assert info["detector"]["model_id"] == "facebook/detr-resnet-50"
    assert info["depth"]["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
