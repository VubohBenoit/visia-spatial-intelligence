"""Deterministic tests for the spatial reasoning layer (no models required)."""

import numpy as np
import pytest

from visia_spatial_intelligence.config import ReasoningConfig
from visia_spatial_intelligence.schemas import (
    BoundingBox,
    CorridorRegion,
    DepthMap,
    Detection,
    RiskLevel,
)
from visia_spatial_intelligence.spatial.reasoning import (
    SpatialReasoner,
    bottom_center_in_corridor,
    classify_risk,
    corridor_overlap,
    relative_proximity,
)

W, H = 640, 480
IN_CORRIDOR = BoundingBox(290, 280, 350, 470)  # entirely below the corridor's top edge (y=0.55)
OUTSIDE = BoundingBox(20, 250, 60, 470)


def _det(box: BoundingBox, track_id: int | None = 1, label: str = "person") -> Detection:
    return Detection(label=label, score=0.9, box=box, track_id=track_id)


def _depth(value: float) -> DepthMap:
    return DepthMap(values=np.full((H, W), value, dtype=np.float32), height=H, width=W)


def test_corridor_overlap_is_full_inside_and_zero_outside() -> None:
    corridor = CorridorRegion()
    assert corridor_overlap(IN_CORRIDOR, corridor, W, H) == pytest.approx(1.0)
    assert corridor_overlap(OUTSIDE, corridor, W, H) == 0.0


def test_bottom_centre_check() -> None:
    corridor = CorridorRegion()
    assert bottom_center_in_corridor(IN_CORRIDOR, corridor, W, H)
    assert not bottom_center_in_corridor(OUTSIDE, corridor, W, H)


def test_relative_proximity_reads_median_inside_box() -> None:
    values = np.zeros((H, W), dtype=np.float32)
    values[250:470, 290:350] = 0.8
    depth = DepthMap(values=values, height=H, width=W)
    assert relative_proximity(IN_CORRIDOR, depth) == pytest.approx(0.8)


def test_classify_risk_outside_corridor_is_none() -> None:
    risk, reasons = classify_risk(0.0, False, 0.9, ReasoningConfig())
    assert risk is RiskLevel.NONE
    assert "outside" in reasons[0]


def test_classify_risk_thresholds_with_depth() -> None:
    config = ReasoningConfig(proximity_high=0.6, proximity_medium=0.35)
    assert classify_risk(1.0, True, 0.7, config)[0] is RiskLevel.HIGH
    assert classify_risk(1.0, True, 0.5, config)[0] is RiskLevel.MEDIUM
    assert classify_risk(1.0, True, 0.1, config)[0] is RiskLevel.LOW


def test_classify_risk_without_depth_is_capped_at_low() -> None:
    risk, reasons = classify_risk(1.0, True, None, ReasoningConfig())
    assert risk is RiskLevel.LOW
    assert any("depth unavailable" in r for r in reasons)


def test_untracked_detection_never_alerts_even_when_high_risk() -> None:
    reasoner = SpatialReasoner(ReasoningConfig(min_frames_for_alert=1))
    assessments, alerts = reasoner.assess([_det(IN_CORRIDOR, track_id=None)], 0, W, H, _depth(0.9))
    assert assessments[0].risk is RiskLevel.HIGH
    assert alerts == []  # no temporal evidence for an untracked object
    assert "no track ID; persistence cannot be established" in assessments[0].reasons


def test_alert_requires_consecutive_persistence() -> None:
    config = ReasoningConfig(min_frames_for_alert=3, alert_cooldown_frames=0)
    reasoner = SpatialReasoner(config)
    depth = _depth(0.9)
    alerts_per_frame = [
        reasoner.assess([_det(IN_CORRIDOR)], frame, W, H, depth)[1] for frame in range(3)
    ]
    assert [len(a) for a in alerts_per_frame] == [0, 0, 1]
    assert alerts_per_frame[2][0].track_id == 1


def test_gap_resets_persistence_streak() -> None:
    config = ReasoningConfig(min_frames_for_alert=2, alert_cooldown_frames=0)
    reasoner = SpatialReasoner(config)
    depth = _depth(0.9)
    reasoner.assess([_det(IN_CORRIDOR)], 0, W, H, depth)
    reasoner.assess([], 1, W, H, depth)  # object disappears for one frame
    assessments, alerts = reasoner.assess([_det(IN_CORRIDOR)], 2, W, H, depth)
    assert assessments[0].frames_observed == 1
    assert alerts == []


def test_cooldown_suppresses_repeat_alerts_for_same_track() -> None:
    config = ReasoningConfig(min_frames_for_alert=1, alert_cooldown_frames=5)
    reasoner = SpatialReasoner(config)
    depth = _depth(0.9)
    emitted = []
    for frame in range(8):
        _, alerts = reasoner.assess([_det(IN_CORRIDOR)], frame, W, H, depth)
        emitted.extend(a.frame_index for a in alerts)
    assert emitted == [0, 5]


def test_outside_corridor_object_is_assessed_but_never_alerts() -> None:
    config = ReasoningConfig(min_frames_for_alert=1, alert_cooldown_frames=0)
    reasoner = SpatialReasoner(config)
    for frame in range(4):
        assessments, alerts = reasoner.assess([_det(OUTSIDE)], frame, W, H, _depth(0.99))
        assert alerts == []
    assert assessments[0].risk is RiskLevel.NONE
    assert assessments[0].frames_observed == 4


def test_reset_clears_streaks_and_cooldowns() -> None:
    config = ReasoningConfig(min_frames_for_alert=1, alert_cooldown_frames=100)
    reasoner = SpatialReasoner(config)
    depth = _depth(0.9)
    assert reasoner.assess([_det(IN_CORRIDOR)], 0, W, H, depth)[1]
    reasoner.reset()
    assert reasoner.assess([_det(IN_CORRIDOR)], 1, W, H, depth)[1]
