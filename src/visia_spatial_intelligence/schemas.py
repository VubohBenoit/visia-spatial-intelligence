"""Typed data structures shared by every pipeline stage.

All coordinates are in pixels of the *original* input image, with the origin at
the top-left corner. Every structure can be serialised with ``to_dict`` so that
intermediate predictions can be inspected and exported as JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RiskLevel(StrEnum):
    """Qualitative risk bucket produced by the spatial reasoning layer."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box ``(x1, y1, x2, y2)`` in pixels, ``x1 <= x2`` and ``y1 <= y2``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bounding box ordering: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersection_area(self, other: BoundingBox) -> float:
        """Area of overlap with ``other`` (zero when disjoint)."""
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        if w <= 0 or h <= 0:
            return 0.0
        return w * h

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other``."""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in a single frame."""

    label: str
    score: float
    box: BoundingBox
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "box": self.box.to_dict(),
            "track_id": self.track_id,
        }


@dataclass(frozen=True, slots=True)
class DepthMap:
    """Relative inverse-depth map (higher value = closer), normalised to ``[0, 1]``.

    This is *relative* depth. It is not calibrated to metres and must never be
    reported as a distance without a validated calibration procedure.
    """

    values: (
        Any  # numpy.ndarray of shape (H, W), float32; kept untyped to avoid importing numpy here
    )
    height: int
    width: int

    def to_dict(self) -> dict[str, Any]:
        return {"height": self.height, "width": self.width, "unit": "relative_inverse_depth"}


@dataclass(frozen=True, slots=True)
class CorridorRegion:
    """Walking-corridor polygon as a trapezoid, in normalised image coordinates ``[0, 1]``.

    ``bottom_*`` is the corridor width at the bottom edge of the image (closest to the
    viewer) and ``top_*`` the width at ``top_y`` (further away). The defaults describe a
    centred corridor for a camera held at roughly waist height.
    """

    bottom_left: float = 0.25
    bottom_right: float = 0.75
    top_left: float = 0.40
    top_right: float = 0.60
    top_y: float = 0.55

    def contains_point(self, x: float, y: float) -> bool:
        """Return whether a normalised point lies inside the trapezoid."""
        if y < self.top_y or y > 1.0:
            return False
        t = (y - self.top_y) / (1.0 - self.top_y)
        left = self.top_left + t * (self.bottom_left - self.top_left)
        right = self.top_right + t * (self.bottom_right - self.top_right)
        return left <= x <= right

    def to_polygon(self, width: int, height: int) -> list[tuple[float, float]]:
        """Return the four corners in pixel coordinates, clockwise from top-left."""
        return [
            (self.top_left * width, self.top_y * height),
            (self.top_right * width, self.top_y * height),
            (self.bottom_right * width, height),
            (self.bottom_left * width, height),
        ]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObstacleAssessment:
    """Interpretable reasoning record for one detected object.

    Every field is kept so that an alert can be explained. ``risk`` is a heuristic
    priority, not a measured probability of collision.
    """

    track_id: int | None
    label: str
    score: float
    corridor_overlap: float
    bottom_center_in_corridor: bool
    relative_proximity: float | None
    frames_observed: int
    risk: RiskLevel
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "score": round(self.score, 4),
            "corridor_overlap": round(self.corridor_overlap, 4),
            "bottom_center_in_corridor": self.bottom_center_in_corridor,
            "relative_proximity": (
                None if self.relative_proximity is None else round(self.relative_proximity, 4)
            ),
            "frames_observed": self.frames_observed,
            "risk": self.risk.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class Alert:
    """An alert emitted after temporal filtering and cooldown checks."""

    frame_index: int
    track_id: int | None
    label: str
    risk: RiskLevel
    message: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "label": self.label,
            "risk": self.risk.value,
            "message": self.message,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class Timings:
    """Wall-clock timings in milliseconds for one processing call."""

    preprocess_ms: float = 0.0
    detection_ms: float = 0.0
    depth_ms: float = 0.0
    tracking_ms: float = 0.0
    reasoning_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass(slots=True)
class FrameResult:
    """Everything the pipeline produced for one image or video frame."""

    frame_index: int
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    assessments: list[ObstacleAssessment] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    disabled_features: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "detections": [d.to_dict() for d in self.detections],
            "assessments": [a.to_dict() for a in self.assessments],
            "alerts": [a.to_dict() for a in self.alerts],
            "timings_ms": self.timings.to_dict(),
            "disabled_features": dict(self.disabled_features),
        }
