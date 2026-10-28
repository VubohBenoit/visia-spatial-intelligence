"""End-to-end analysis pipeline.

Order of stages::

    validation -> detection -> tracking (optional) -> depth (optional)
              -> spatial reasoning -> alerts -> timings

Detection is required: if the detector cannot be loaded, :class:`ModelUnavailableError`
propagates to the caller. Tracking and depth are optional. When one of them is missing
or fails, it is recorded in ``FrameResult.disabled_features`` and the remaining stages
run without it. No placeholder output is ever generated for a disabled feature.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from visia_spatial_intelligence.config import AppConfig
from visia_spatial_intelligence.depth.estimator import DepthAnythingEstimator
from visia_spatial_intelligence.detection.detector import (
    DetrDetector,
    ModelUnavailableError,
    ObjectDetector,
)
from visia_spatial_intelligence.schemas import DepthMap, FrameResult, Timings
from visia_spatial_intelligence.spatial.reasoning import SpatialReasoner
from visia_spatial_intelligence.tracking.tracker import IoUTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisOutput:
    """Result of analysing one frame, including the depth map used for visualisation."""

    result: FrameResult
    depth: DepthMap | None = None


@dataclass(slots=True)
class VisiaPipeline:
    """Coordinates the stages. Build with :meth:`from_config` in normal use."""

    config: AppConfig
    detector: ObjectDetector | None
    depth_estimator: DepthAnythingEstimator | None
    tracker: IoUTracker | None
    reasoner: SpatialReasoner
    _disabled_at_build: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: AppConfig) -> VisiaPipeline:
        """Create the pipeline. Models are loaded lazily on first use, not here."""
        detector = DetrDetector(config.detector)
        depth = DepthAnythingEstimator(config.depth) if config.depth.enabled else None
        tracker = IoUTracker(config.tracker) if config.tracker.enabled else None
        disabled: dict[str, str] = {}
        if depth is None:
            disabled["depth"] = "disabled by configuration"
        if tracker is None:
            disabled["tracking"] = "disabled by configuration"
        return cls(
            config=config,
            detector=detector,
            depth_estimator=depth,
            tracker=tracker,
            reasoner=SpatialReasoner(config.reasoning),
            _disabled_at_build=disabled,
        )

    def reset(self) -> None:
        """Clear temporal state before processing a new video."""
        if self.tracker is not None:
            self.tracker.reset()
        self.reasoner.reset()

    def analyze(self, image: Image.Image, frame_index: int = 0) -> AnalysisOutput:
        """Analyse one RGB image.

        Raises:
            ModelUnavailableError: if the required detector cannot be loaded.
        """
        if self.detector is None:
            raise ModelUnavailableError("No object detector is configured")

        width, height = image.size
        disabled = dict(self._disabled_at_build)
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        detections = self.detector.detect(image)
        timings["detection_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        if self.tracker is not None:
            detections = self.tracker.update(detections)
        timings["tracking_ms"] = _elapsed_ms(t0)

        depth: DepthMap | None = None
        t0 = time.perf_counter()
        if self.depth_estimator is not None:
            try:
                depth = self.depth_estimator.estimate(image)
            except Exception as exc:  # depth is optional: degrade instead of failing the frame
                disabled["depth"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Depth disabled for this frame")
        timings["depth_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        assessments, alerts = self.reasoner.assess(
            detections, frame_index=frame_index, width=width, height=height, depth=depth
        )
        timings["reasoning_ms"] = _elapsed_ms(t0)
        timings["total_ms"] = _elapsed_ms(total_start)

        result = FrameResult(
            frame_index=frame_index,
            width=width,
            height=height,
            detections=detections,
            assessments=assessments,
            alerts=alerts,
            timings=Timings(**timings),
            disabled_features=disabled,
        )
        return AnalysisOutput(result=result, depth=depth)

    def describe(self) -> dict[str, Any]:
        """Return the model and feature configuration used by this pipeline, for reports."""
        detector_cfg = self.config.detector
        depth_cfg = self.config.depth
        return {
            "detector": {
                "model_id": detector_cfg.model_id,
                "revision": detector_cfg.revision,
                "score_threshold": detector_cfg.score_threshold,
                "classes": list(detector_cfg.classes),
            },
            "depth": {
                "enabled": depth_cfg.enabled,
                "model_id": depth_cfg.model_id,
                "revision": depth_cfg.revision,
            },
            "tracker": {
                "enabled": self.tracker is not None,
                "iou_threshold": self.config.tracker.iou_threshold,
                "max_missed_frames": self.config.tracker.max_missed_frames,
            },
            "reasoning": {
                "min_frames_for_alert": self.config.reasoning.min_frames_for_alert,
                "alert_cooldown_frames": self.config.reasoning.alert_cooldown_frames,
                "corridor": self.config.reasoning.corridor.to_dict(),
            },
            "seed": self.config.seed,
        }


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
