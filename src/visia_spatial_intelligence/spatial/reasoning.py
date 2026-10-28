"""Interpretable spatial reasoning over detections, depth and a walking corridor.

The layer answers three questions for every detected object, each with a recorded reason:

1. *Is it in the walking corridor?* Measured by the fraction of a grid of points inside
   the box that fall within the corridor trapezoid, and by whether the bottom-centre
   point (where an object meets the ground) is inside it.
2. *How close is it, relatively?* The median relative inverse depth inside the box, if a
   depth map is available. This is a relative ordering, not a distance in metres.
3. *Has it persisted long enough to alert?* Consecutive frames with the same track ID.

Bounding-box overlap and monocular depth are treated as *evidence*, not proof. Alerts
require both a non-trivial risk level and temporal persistence, and each track has a
cooldown so that one object does not repeatedly trigger notifications.
"""

from __future__ import annotations

import numpy as np

from visia_spatial_intelligence.config import ReasoningConfig
from visia_spatial_intelligence.schemas import (
    Alert,
    BoundingBox,
    CorridorRegion,
    DepthMap,
    Detection,
    ObstacleAssessment,
    RiskLevel,
)

_GRID = 8  # points per axis used to estimate corridor overlap


def corridor_overlap(box: BoundingBox, corridor: CorridorRegion, width: int, height: int) -> float:
    """Fraction of a regular grid of points inside ``box`` that lie within the corridor."""
    xs = _sample_axis(box.x1, box.x2, width)
    ys = _sample_axis(box.y1, box.y2, height)
    inside = 0
    for y in ys:
        for x in xs:
            if corridor.contains_point(x, y):
                inside += 1
    return inside / (len(xs) * len(ys))


def _sample_axis(low: float, high: float, size: int) -> list[float]:
    """Evenly spaced sample positions along one axis, normalised to ``[0, 1]``."""
    return [float(v) / size for v in np.linspace(low, high, _GRID)]


def bottom_center_in_corridor(
    box: BoundingBox, corridor: CorridorRegion, width: int, height: int
) -> bool:
    """Whether the point where the object meets the ground lies inside the corridor."""
    cx = (box.x1 + box.x2) / 2.0 / width
    by = box.y2 / height
    return corridor.contains_point(cx, min(by, 1.0))


def relative_proximity(box: BoundingBox, depth: DepthMap) -> float:
    """Median normalised inverse depth inside the box; higher means relatively closer."""
    x1 = int(np.clip(np.floor(box.x1), 0, depth.width - 1))
    x2 = int(np.clip(np.ceil(box.x2), x1 + 1, depth.width))
    y1 = int(np.clip(np.floor(box.y1), 0, depth.height - 1))
    y2 = int(np.clip(np.ceil(box.y2), y1 + 1, depth.height))
    region = np.asarray(depth.values)[y1:y2, x1:x2]
    return float(np.median(region))


def classify_risk(
    overlap: float,
    centre_in_corridor: bool,
    proximity: float | None,
    config: ReasoningConfig,
) -> tuple[RiskLevel, tuple[str, ...]]:
    """Map the evidence for one object to a priority level, with human-readable reasons.

    The output is a heuristic priority ordering. It is not a calibrated probability.
    """
    in_corridor = centre_in_corridor or overlap >= config.min_corridor_overlap
    if not in_corridor:
        return RiskLevel.NONE, ("outside the walking corridor",)

    where = (
        "bottom-centre inside corridor"
        if centre_in_corridor
        else (f"partial corridor overlap ({overlap:.0%})")
    )
    if proximity is None:
        return RiskLevel.LOW, (where, "depth unavailable; risk limited to corridor presence")
    if proximity >= config.proximity_high:
        level = RiskLevel.HIGH
        cmp = f">= high threshold {config.proximity_high:.2f}"
    elif proximity >= config.proximity_medium:
        level = RiskLevel.MEDIUM
        cmp = f">= medium threshold {config.proximity_medium:.2f}"
    else:
        level = RiskLevel.LOW
        cmp = f"< medium threshold {config.proximity_medium:.2f}"
    return level, (where, f"relative depth {proximity:.2f} {cmp}")


class SpatialReasoner:
    """Stateful reasoner that tracks streaks and alert cooldowns across frames."""

    def __init__(self, config: ReasoningConfig) -> None:
        self._config = config
        self._streaks: dict[
            int, tuple[int, int]
        ] = {}  # track_id -> (consecutive frames, last frame)
        self._last_alert: dict[str, int] = {}  # alert key -> frame index of last alert

    def reset(self) -> None:
        """Clear temporal state, e.g. when a new video starts."""
        self._streaks.clear()
        self._last_alert.clear()

    def assess(
        self,
        detections: list[Detection],
        frame_index: int,
        width: int,
        height: int,
        depth: DepthMap | None = None,
    ) -> tuple[list[ObstacleAssessment], list[Alert]]:
        """Assess every detection in one frame and emit any alerts that pass the filters.

        ``frame_index`` must be non-decreasing across calls.
        """
        corridor = self._config.corridor
        assessments: list[ObstacleAssessment] = []
        alerts: list[Alert] = []

        for det in detections:
            frames = self._update_streak(det.track_id, frame_index)
            overlap = corridor_overlap(det.box, corridor, width, height)
            centre_in = bottom_center_in_corridor(det.box, corridor, width, height)
            proximity = relative_proximity(det.box, depth) if depth is not None else None
            risk, reasons = classify_risk(overlap, centre_in, proximity, self._config)

            if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH):
                if det.track_id is None:
                    reasons = (*reasons, "no track ID; persistence cannot be established")
                elif frames < self._config.min_frames_for_alert:
                    reasons = (
                        *reasons,
                        f"only {frames} consecutive frame(s); "
                        f"{self._config.min_frames_for_alert} required before alerting",
                    )
                else:
                    alert = self._maybe_alert(det, risk, reasons, frame_index)
                    if alert is not None:
                        alerts.append(alert)
                        reasons = (*reasons, "alert issued")
                    else:
                        reasons = (*reasons, "suppressed by alert cooldown")

            assessments.append(
                ObstacleAssessment(
                    track_id=det.track_id,
                    label=det.label,
                    score=det.score,
                    corridor_overlap=overlap,
                    bottom_center_in_corridor=centre_in,
                    relative_proximity=proximity,
                    frames_observed=frames,
                    risk=risk,
                    reasons=reasons,
                )
            )

        self._forget_stale_streaks(frame_index)
        return assessments, alerts

    def _update_streak(self, track_id: int | None, frame_index: int) -> int:
        if track_id is None:
            return 1  # untracked detections carry no temporal evidence
        count, last = self._streaks.get(track_id, (0, -2))
        count = count + 1 if last >= frame_index - 1 else 1
        self._streaks[track_id] = (count, frame_index)
        return count

    def _forget_stale_streaks(self, frame_index: int) -> None:
        stale = [tid for tid, (_, last) in self._streaks.items() if last < frame_index]
        for tid in stale:
            del self._streaks[tid]

    def _maybe_alert(
        self,
        det: Detection,
        risk: RiskLevel,
        reasons: tuple[str, ...],
        frame_index: int,
    ) -> Alert | None:
        key = f"track:{det.track_id}" if det.track_id is not None else f"label:{det.label}"
        last = self._last_alert.get(key)
        if last is not None and frame_index - last < self._config.alert_cooldown_frames:
            return None
        self._last_alert[key] = frame_index
        return Alert(
            frame_index=frame_index,
            track_id=det.track_id,
            label=det.label,
            risk=risk,
            message=f"{det.label.capitalize()} in walking corridor (priority: {risk.value})",
            reasons=reasons,
        )
