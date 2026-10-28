"""Monocular relative depth estimation.

:class:`DepthAnythingEstimator` wraps ``depth-anything/Depth-Anything-V2-Small-hf``
(Apache-2.0, see docs/LICENSE_AUDIT.md). The model predicts *relative inverse depth*:
larger values are closer to the camera. Values are normalised per image to ``[0, 1]``
and must not be converted into metric distances without a calibration procedure.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

from visia_spatial_intelligence.config import DepthConfig
from visia_spatial_intelligence.detection.detector import ModelUnavailableError
from visia_spatial_intelligence.schemas import DepthMap

logger = logging.getLogger(__name__)


class DepthAnythingEstimator:
    """Relative depth estimator with lazy model loading."""

    def __init__(self, config: DepthConfig) -> None:
        self._config = config
        self._processor: object | None = None
        self._model: object | None = None
        self.load_time_ms: float = 0.0

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def load(self) -> None:
        """Download (once) and load the checkpoint. Raises :class:`ModelUnavailableError`."""
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ModelUnavailableError(f"transformers/torch not installed: {exc}") from exc

        start = time.perf_counter()
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self._config.model_id, revision=self._config.revision
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                self._config.model_id, revision=self._config.revision
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load depth model '{self._config.model_id}': {exc}"
            ) from exc
        model.eval()
        self._model = model
        self.load_time_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Loaded depth model %s in %.0f ms", self._config.model_id, self.load_time_ms)

    def estimate(self, image: Image.Image) -> DepthMap:
        """Return a relative inverse-depth map with the same resolution as ``image``."""
        self.load()
        assert self._processor is not None and self._model is not None
        import torch
        import torch.nn.functional as functional

        with torch.inference_mode():
            inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[operator]
            outputs = self._model(**inputs)  # type: ignore[operator]
            predicted = outputs.predicted_depth.unsqueeze(1)
            resized = functional.interpolate(
                predicted, size=(image.height, image.width), mode="bilinear", align_corners=False
            )[0, 0]

        values = resized.float().cpu().numpy()
        values = _normalise(values)
        return DepthMap(values=values, height=image.height, width=image.width)


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max normalise to ``[0, 1]``; a constant map becomes all zeros."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)
