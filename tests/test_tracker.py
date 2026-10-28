"""Deterministic tests for the IoU tracker."""

from visia_spatial_intelligence.config import TrackerConfig
from visia_spatial_intelligence.schemas import BoundingBox, Detection
from visia_spatial_intelligence.tracking.tracker import IoUTracker


def _det(label: str, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(label=label, score=0.9, box=BoundingBox(x1, y1, x2, y2))


def test_same_object_keeps_its_id_across_frames() -> None:
    tracker = IoUTracker(TrackerConfig())
    first = tracker.update([_det("person", 10, 10, 50, 90)])
    second = tracker.update([_det("person", 12, 11, 52, 91)])
    assert first[0].track_id == second[0].track_id == 1


def test_new_objects_receive_new_ids() -> None:
    tracker = IoUTracker(TrackerConfig())
    out = tracker.update([_det("car", 0, 0, 40, 40), _det("car", 200, 0, 240, 40)])
    assert [d.track_id for d in out] == [1, 2]


def test_different_labels_never_match() -> None:
    tracker = IoUTracker(TrackerConfig())
    tracker.update([_det("person", 10, 10, 50, 90)])
    out = tracker.update([_det("bicycle", 10, 10, 50, 90)])
    assert out[0].track_id == 2


def test_track_survives_short_gap_but_not_long_gap() -> None:
    tracker = IoUTracker(TrackerConfig(max_missed_frames=2))
    box = [_det("person", 10, 10, 50, 90)]
    assert tracker.update(box)[0].track_id == 1
    tracker.update([])  # one missed frame
    tracker.update([])  # two missed frames
    assert tracker.update(box)[0].track_id == 1  # still within tolerance
    for _ in range(3):
        tracker.update([])
    assert tracker.update(box)[0].track_id == 2  # dropped; a new ID is issued


def test_ids_are_not_reused_after_reset_is_not_called() -> None:
    tracker = IoUTracker(TrackerConfig(max_missed_frames=0))
    tracker.update([_det("person", 0, 0, 30, 60)])
    tracker.update([])
    out = tracker.update([_det("person", 0, 0, 30, 60)])
    assert out[0].track_id == 2


def test_reset_restarts_numbering() -> None:
    tracker = IoUTracker(TrackerConfig())
    tracker.update([_det("person", 0, 0, 30, 60)])
    tracker.reset()
    assert tracker.update([_det("person", 0, 0, 30, 60)])[0].track_id == 1


def test_low_overlap_below_threshold_starts_new_track() -> None:
    tracker = IoUTracker(TrackerConfig(iou_threshold=0.5))
    tracker.update([_det("person", 0, 0, 100, 100)])
    out = tracker.update([_det("person", 60, 60, 160, 160)])  # IoU ~0.2
    assert out[0].track_id == 2
