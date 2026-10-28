"""Object detection behind a replaceable interface.

:class:`DetrDetector` wraps ``facebook/detr-resnet-50`` (Apache-2.0, see
docs/LICENSE_AUDIT.md) through Hugging Face ``transformers``. Model loading is lazy so
that importing the package and running unit tests never downloads weights.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from PIL import Image

from visia_spatial_intelligence.config import DetectorConfig
from visia_spatial_intelligence.schemas import BoundingBox, Detection

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """A required model could not be loaded (missing cache, no network, bad revision...)."""


class ObjectDetector(Protocol):
    """Anything that maps an RGB image to a list of :class:`Detection` objects."""

    def detect(self, image: Image.Image) -> list[Detection]: ...


class DetrDetector:
    """DETR object detector restricted to the configured navigation classes."""

    def __init__(self, config: DetectorConfig) -> None:
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
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise ModelUnavailableError(f"transformers/torch not installed: {exc}") from exc

        start = time.perf_counter()
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self._config.model_id, revision=self._config.revision
            )
            model = AutoModelForObjectDetection.from_pretrained(
                self._config.model_id, revision=self._config.revision
            )
        except Exception as exc:  # transformers raises several exception types
            raise ModelUnavailableError(
                f"Could not load detector '{self._config.model_id}': {exc}"
            ) from exc
        model.eval()
        self._model = model
        self.load_time_ms = (time.perf_counter() - start) * 1000.0
        logger.info("Loaded detector %s in %.0f ms", self._config.model_id, self.load_time_ms)

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run detection on an RGB image. Boxes are returned in original-pixel coordinates."""
        self.load()
        assert self._processor is not None and self._model is not None
        import torch

        target_sizes = torch.tensor([[image.height, image.width]])
        with torch.inference_mode():
            inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[operator]
            outputs = self._model(**inputs)  # type: ignore[operator]
            results = self._processor.post_process_object_detection(  # type: ignore[attr-defined]
                outputs, threshold=self._config.score_threshold, target_sizes=target_sizes
            )[0]
        id2label: dict[int, str] = self._model.config.id2label  # type: ignore[attr-defined]
        allowed = set(self._config.classes)

        detections: list[Detection] = []
        for score, label_id, box in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
            strict=True,
        ):
            label = id2label.get(int(label_id), f"class_{label_id}")
            if label not in allowed:
                continue
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    label=label,
                    score=float(score),
                    box=BoundingBox(*_clip_box(x1, y1, x2, y2, image.width, image.height)),
                )
            )
        return detections


def _clip_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Clamp a box to the image bounds and keep ordering valid."""
    x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
    y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
    return x1, y1, x2, y2
