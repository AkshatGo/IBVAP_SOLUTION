"""
Object Tracker — self-contained multi-object tracker with persistent IDs.

Earlier versions of this module tried to delegate to an external
`byte_tracker` package or `ultralytics.trackers.ByteTrack`. Both fail
silently in practice: the `byte_tracker` PyPI package isn't installed by
default, and different `ultralytics` releases have shipped incompatible
tracker class names/constructor signatures (`ByteTrack` vs `BYTETracker`,
different `.update()` call shapes). When both imports fail, the old code
fell back to "assign sequential IDs every frame" — which is not a tracker
at all, it's a no-op that silently breaks everything downstream that
depends on a *stable* track_id: fence entry/exit detection, speed/bearing
estimation, and ANPR multi-frame consensus voting.

This module implements a small, dependency-free greedy IoU tracker instead
(single-stage association + track aging, in the spirit of ByteTrack without
its two-stage low/high-confidence matching machinery). It has no optional
imports and therefore always works the same way regardless of what's
installed.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from .detector import Detection


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: tuple
    class_id: int
    class_name: str
    confidence: float
    hits: int = 1
    time_since_update: int = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= 1  # see ObjectTracker.min_hits for the real gate


class ObjectTracker:
    """
    Multi-object tracker with persistent IDs across frames.

    Association strategy: greedy IoU matching between each active track's
    last-known box and the current frame's detections (same class only),
    highest-IoU-first, above `match_thresh`. Unmatched tracks age out after
    `track_buffer` consecutive missed frames; unmatched detections become
    new tracks immediately (returned track_ids are stable and monotonically
    increasing — never reused).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_hits: int = 3,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1
        self.frame_id = 0
        self.track_history: Dict[int, List[tuple]] = defaultdict(list)

    def update(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        """
        Update tracker with new detections. Returns the same detections with
        `track_id` filled in with a persistent ID (stable across frames for
        the same physical object, best-effort).
        """
        self.frame_id += 1
        dets = [d for d in detections if d.confidence >= self.track_thresh]

        # Greedy IoU association, same-class only
        candidates = []
        for tid, tr in self._tracks.items():
            for i, d in enumerate(dets):
                if tr.class_id != d.class_id:
                    continue
                iou = _iou(tr.bbox, d.bbox)
                if iou >= self.match_thresh:
                    candidates.append((iou, tid, i))
        candidates.sort(key=lambda x: -x[0])

        matched_tracks = set()
        matched_dets = set()
        for iou, tid, i in candidates:
            if tid in matched_tracks or i in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(i)

            tr = self._tracks[tid]
            d = dets[i]
            tr.bbox = d.bbox
            tr.confidence = d.confidence
            tr.hits += 1
            tr.time_since_update = 0

            d.track_id = tid
            self.track_history[tid].append(d.center)

        # Age out tracks that went unmatched this frame
        for tid in list(self._tracks.keys()):
            if tid in matched_tracks:
                continue
            tr = self._tracks[tid]
            tr.time_since_update += 1
            if tr.time_since_update > self.track_buffer:
                del self._tracks[tid]

        # Spawn new tracks for unmatched detections
        for i, d in enumerate(dets):
            if i in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid, bbox=d.bbox, class_id=d.class_id,
                class_name=d.class_name, confidence=d.confidence,
            )
            d.track_id = tid
            self.track_history[tid].append(d.center)

        return [d for d in dets if d.track_id != -1]

    def get_trajectory(self, track_id: int) -> List[tuple]:
        """Get full trajectory for a tracked object."""
        return list(self.track_history.get(track_id, []))

    def get_speed(self, track_id: int, fps: float = 30.0) -> float:
        """Estimate speed in pixels/second from recent trajectory."""
        traj = self.track_history.get(track_id, [])
        if len(traj) < 2:
            return 0.0
        recent = traj[-10:]  # last 10 positions
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        time_span = len(recent) / fps
        return dist / max(time_span, 1e-6)

    def get_bearing(self, track_id: int) -> str:
        """Get compass bearing from trajectory."""
        traj = self.track_history.get(track_id, [])
        if len(traj) < 2:
            return "unknown"
        dx = traj[-1][0] - traj[-2][0]
        dy = traj[-1][1] - traj[-2][1]
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "S" if dy > 0 else "N"

    def is_confirmed(self, track_id: int) -> bool:
        """True once a track has been matched enough times to trust (reduces flicker)."""
        tr = self._tracks.get(track_id)
        return tr is not None and tr.hits >= self.min_hits

    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self):
        """Reset tracker state."""
        self.frame_id = 0
        self._tracks.clear()
        self._next_id = 1
        self.track_history.clear()
