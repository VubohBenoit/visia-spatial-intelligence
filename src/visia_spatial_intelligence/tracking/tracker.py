"""Greedy IoU tracker for associating detections across consecutive frames.

Algorithm (a simplified SORT-style association with no motion model):

1. Within each class label, compute the IoU between every active track's last box and
   every new detection.
2. Greedily match the highest-IoU pairs whose IoU is at least ``iou_threshold``.
3. Matched tracks keep their ID and update their box. Unmatched detections start new
   tracks with fresh, monotonically increasing IDs.
4. Unmatched tracks accumulate a miss count and are dropped after
   ``max_missed_frames`` consecutive misses. Dropped IDs are never reused.

Known failure modes (documented in docs/LIMITATIONS.md):

* No motion model, so fast-moving objects can fall below the IoU threshold and receive
  a new ID (identity switch).
* Heavy occlusion longer than ``max_missed_frames`` ends a track permanently.
* Two nearby same-class objects can swap IDs when their boxes overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from visia_spatial_intelligence.config import TrackerConfig
from visia_spatial_intelligence.schemas import BoundingBox, Detection


@dataclass(slots=True)
class _Track:
    track_id: int
    label: str
    box: BoundingBox
    missed: int = 0


class IoUTracker:
    """Stateful tracker; call :meth:`update` once per frame in temporal order."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Forget all tracks, e.g. at the start of a new video."""
        self._tracks = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign a ``track_id`` to every detection and return the updated list."""
        matched_tracks: set[int] = set()
        assigned: dict[int, int] = {}  # detection index -> track id

        candidates: list[tuple[float, int, int]] = []  # (iou, detection index, track index)
        for d_idx, det in enumerate(detections):
            for t_idx, track in enumerate(self._tracks):
                if track.label != det.label:
                    continue
                overlap = det.box.iou(track.box)
                if overlap >= self._config.iou_threshold:
                    candidates.append((overlap, d_idx, t_idx))

        # Highest IoU first; ties broken by index for determinism.
        for _, d_idx, t_idx in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
            if d_idx in assigned or t_idx in matched_tracks:
                continue
            track = self._tracks[t_idx]
            track.box = detections[d_idx].box
            track.missed = 0
            assigned[d_idx] = track.track_id
            matched_tracks.add(t_idx)

        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                track.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self._config.max_missed_frames]

        output: list[Detection] = []
        for d_idx, det in enumerate(detections):
            track_id = assigned.get(d_idx)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks.append(_Track(track_id=track_id, label=det.label, box=det.box))
            output.append(replace(det, track_id=track_id))
        return output
