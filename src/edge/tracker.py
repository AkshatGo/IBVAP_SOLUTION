"""
Object Tracker — wraps ByteTrack for persistent object IDs across frames.
Assigns stable track_ids to detections so we can follow objects over time.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from .detector import Detection

try:
    from byte_tracker.byte_tracker import ByteTracker as _ByteTracker
except ImportError:
    try:
        from ultralytics.trackers import ByteTrack as _ByteTracker
    except ImportError:
        _ByteTracker = None


class ObjectTracker:
    """
    Multi-object tracker using ByteTrack.
    Maintains persistent IDs for each tracked object across frames.
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
        self.tracker = None
        self.frame_id = 0
        self.track_history: Dict[int, List[tuple]] = defaultdict(list)

    def _init_tracker(self):
        """Initialize the ByteTrack tracker."""
        if _ByteTracker is not None:
            self.tracker = _ByteTracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
            )

    def update(self, detections: List[Detection], frame_shape: tuple) -> List[Detection]:
        """
        Update tracker with new detections.
        Returns detections with assigned track_ids.
        """
        self.frame_id += 1

        if self.tracker is None:
            self._init_tracker()

        if self.tracker is None or not detections:
            # Fallback: assign sequential IDs
            for i, det in enumerate(detections):
                det.track_id = i
                self.track_history[det.track_id].append(det.center)
            return detections

        # Convert detections to [x1, y1, x2, y2, conf] format for ByteTrack
        dets_np = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
            for d in detections
        ], dtype=np.float32)

        if len(dets_np) == 0:
            return detections

        # Run tracker
        online_targets = self.tracker.update(dets_np, frame_shape, frame_shape)

        # Map tracked IDs back to detections
        tracked = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            conf = t.score if hasattr(t, 'score') else 0.0

            # Find matching detection by IoU
            best_match = None
            best_iou = 0.0
            tx1, ty1 = tlwh[0], tlwh[1]
            tx2, ty2 = tx1 + tlwh[2], ty1 + tlwh[3]

            for det in detections:
                ix1 = max(det.bbox[0], tx1)
                iy1 = max(det.bbox[1], ty1)
                ix2 = min(det.bbox[2], tx2)
                iy2 = min(det.bbox[3], ty2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
                area_b = tlwh[2] * tlwh[3]
                iou = inter / max(area_a + area_b - inter, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det

            if best_match is not None:
                best_match.track_id = tid
                self.track_history[tid].append(best_match.center)
                tracked.append(best_match)

        return tracked

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

    def reset(self):
        """Reset tracker state."""
        self.frame_id = 0
        self.track_history.clear()
        if self.tracker is not None:
            self._init_tracker()
