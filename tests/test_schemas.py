"""Tests for geometric primitives and serialisation."""

import pytest

from visia_spatial_intelligence.schemas import BoundingBox, CorridorRegion, RiskLevel


def test_box_area_and_center() -> None:
    box = BoundingBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.area == 800
    assert box.center == (20, 40)


def test_box_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError, match="Invalid bounding box"):
        BoundingBox(30, 0, 10, 10)


def test_iou_identical_disjoint_and_partial() -> None:
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0
    # Overlap 5x10=50, union 100+100-50=150
    assert a.iou(BoundingBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_corridor_contains_points_as_expected() -> None:
    corridor = CorridorRegion()
    assert corridor.contains_point(0.5, 0.9)
    assert not corridor.contains_point(0.05, 0.9)
    assert not corridor.contains_point(0.5, 0.1)  # above the horizon line


def test_risk_levels_serialise_to_strings() -> None:
    assert RiskLevel.HIGH.value == "high"
    assert RiskLevel("medium") is RiskLevel.MEDIUM
