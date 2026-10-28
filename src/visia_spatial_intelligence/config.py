"""Configuration loading and validation.

Configuration lives in ``configs/default.yaml``. Any key can be overridden by passing
a partial mapping to :func:`load_config`; unknown keys are rejected so that typos
do not silently change behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from visia_spatial_intelligence.schemas import CorridorRegion

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

# COCO class names used by DETR. Only these are treated as navigation-relevant obstacles.
NAVIGATION_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "fire hydrant",
    "dog",
)


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    model_id: str = "facebook/detr-resnet-50"
    revision: str | None = None
    score_threshold: float = 0.5
    classes: tuple[str, ...] = NAVIGATION_CLASSES

    def __post_init__(self) -> None:
        if not 0.0 < self.score_threshold < 1.0:
            raise ConfigError("detector.score_threshold must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class DepthConfig:
    enabled: bool = True
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    enabled: bool = True
    iou_threshold: float = 0.3
    max_missed_frames: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ConfigError("tracker.iou_threshold must be in (0, 1]")
        if self.max_missed_frames < 0:
            raise ConfigError("tracker.max_missed_frames must be >= 0")


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """Thresholds for the interpretable obstacle-risk rules."""

    min_frames_for_alert: int = 3
    alert_cooldown_frames: int = 15
    min_corridor_overlap: float = 0.05
    proximity_high: float = 0.6
    proximity_medium: float = 0.35
    corridor: CorridorRegion = field(default_factory=CorridorRegion)

    def __post_init__(self) -> None:
        if self.min_frames_for_alert < 1:
            raise ConfigError("reasoning.min_frames_for_alert must be >= 1")
        if self.alert_cooldown_frames < 0:
            raise ConfigError("reasoning.alert_cooldown_frames must be >= 0")
        if not self.proximity_medium <= self.proximity_high <= 1.0:
            raise ConfigError("proximity thresholds must satisfy medium <= high <= 1")


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_side: int = 4096
    min_image_side: int = 32
    max_video_frames: int = 300


@dataclass(frozen=True, slots=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    seed: int = 0

    def with_overrides(self, **sections: Any) -> AppConfig:
        """Return a copy with whole sections replaced, e.g. ``depth=DepthConfig(enabled=False)``."""
        valid = {f.name for f in fields(self)}
        unknown = set(sections) - valid
        if unknown:
            raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
        return replace(self, **sections)


def _build_section(cls: type, raw: dict[str, Any] | None, name: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{name}' must be a mapping")
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"Unknown keys in '{name}': {sorted(unknown)}")
    cleaned = dict(raw)
    if "classes" in cleaned:
        cleaned["classes"] = tuple(cleaned["classes"])
    if "corridor" in cleaned:
        cleaned["corridor"] = CorridorRegion(**cleaned["corridor"])
    try:
        return cls(**cleaned)
    except TypeError as exc:
        raise ConfigError(f"Invalid value in '{name}': {exc}") from exc


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML, falling back to built-in defaults for missing sections."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Top-level configuration must be a mapping")
    unknown = set(raw) - {"detector", "depth", "tracker", "reasoning", "limits", "seed"}
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    return AppConfig(
        detector=_build_section(DetectorConfig, raw.get("detector"), "detector"),
        depth=_build_section(DepthConfig, raw.get("depth"), "depth"),
        tracker=_build_section(TrackerConfig, raw.get("tracker"), "tracker"),
        reasoning=_build_section(ReasoningConfig, raw.get("reasoning"), "reasoning"),
        limits=_build_section(LimitsConfig, raw.get("limits"), "limits"),
        seed=int(raw.get("seed", 0)),
    )
