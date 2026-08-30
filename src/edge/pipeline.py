"""
Main Inference Pipeline — Ties all edge modules together.
Processes video frames through the full detection → tracking → analysis → alert pipeline.
"""
import cv2
import numpy as np
import time
import uuid
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

from .detector import EdgeDetector, Detection
from .tracker import ObjectTracker
from .anpr import ANPREngine, ConsensusResult
from .fence import VirtualFence, IntrusionEvent
from .signal import SignalLossDetector
from .hashchain import HashChain, EventRecord
from ..config import CONFIG
from ..utils.logger import log


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_id: int
    timestamp: float
    detections: List[dict]
    tracks: List[dict]
    intrusions: List[dict]
    plates: List[dict]
    alerts: List[dict]
    camera_status: Dict[str, dict]
    annotated_frame: Optional[np.ndarray] = None
    hash_chain_head: str = ""


class IBVAPPipeline:
    """
    Full IBVAP inference pipeline.
    
    Flow:
    1. Frame → YOLOv8 detection (persons, vehicles)
    2. Detections → IoU tracker (persistent IDs)
    3. Tracked objects → Virtual fence check
    4. Vehicle crops → ANPR (plate recognition)
    5. Camera → Signal loss check
    6. Events → Hash chain (tamper-evident log)
    """

    def __init__(self, camera_id: str = "CAM-01", site_id: str = "BOP-01"):
        self.camera_id = camera_id
        self.site_id = site_id

        # Initialize modules from CONFIG, so the detection and plate models
        # can be swapped for fine-tuned checkpoints via IBVAP_DETECTION_MODEL
        # / IBVAP_PLATE_MODEL without touching code (ROADMAP §5.2).
        self.detector = EdgeDetector(
            model_path=CONFIG.detection.model_path,
            confidence=CONFIG.detection.confidence_threshold,
            target_classes=CONFIG.detection.target_classes,
            input_size=CONFIG.detection.input_size,
        )
        self.tracker = ObjectTracker()
        self.anpr = ANPREngine(
            consensus_frames=CONFIG.anpr.consensus_frames,
            min_confidence=CONFIG.anpr.min_confidence,
            ocr_languages=CONFIG.anpr.languages,
            plate_model_path=CONFIG.anpr.plate_model_path,
            plate_model_confidence=CONFIG.anpr.plate_model_confidence,
        )
        self.fence = VirtualFence(cooldown_seconds=CONFIG.fence.alert_cooldown_seconds)
        self.signal_detector = SignalLossDetector(
            timeout_seconds=CONFIG.camera.signal_loss_timeout_seconds
        )
        self.hash_chain = HashChain()

        # State
        self.frame_id = 0
        self.running = False
        self._model_loaded = False

        # Callbacks
        self._on_alert: List[Callable] = []
        self._on_detection: List[Callable] = []

    def load(self):
        """Load all ML models."""
        if not self._model_loaded:
            log.info(f"Loading models for {self.site_id}/{self.camera_id}")
            log.info(f"Detection model: {self.detector.model_path}")
            self.detector.load()
            self.anpr.load()
            self.signal_detector.register_camera(self.camera_id)
            self._model_loaded = True
            log.info("Models loaded successfully")
        return self

    def setup_fence(self, zones: Optional[List[dict]] = None):
        """Set up virtual fence zones."""
        if zones:
            for zone in zones:
                self.fence.add_zone(
                    name=zone["name"],
                    polygon=zone["polygon"],
                    severity=zone.get("severity", "high"),
                    description=zone.get("description", ""),
                )
        else:
            self.fence.add_default_zone()

    def on_alert(self, callback: Callable):
        """Register alert callback."""
        self._on_alert.append(callback)

    def on_detection(self, callback: Callable):
        """Register detection callback."""
        self._on_detection.append(callback)

    def _fire_alert(self, event_type: str, severity: str, payload: dict) -> EventRecord:
        """Create an alert event and add to hash chain."""
        event_id = f"e{uuid.uuid4().hex[:8]}"
        record = self.hash_chain.add_event(
            event_id=event_id,
            event_type=event_type,
            site_id=self.site_id,
            camera_id=self.camera_id,
            severity=severity,
            payload=payload,
        )
        # Notify callbacks
        for cb in self._on_alert:
            try:
                cb(record)
            except Exception:
                pass
        return record

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single video frame through the full pipeline.
        Returns annotated frame and all detection results.
        """
        self.frame_id += 1
        now = time.time()
        alerts = []

        # 1. Signal loss check
        signal_msg = self.signal_detector.update(self.camera_id, frame)
        if signal_msg and "restored" not in signal_msg.lower():
            log.signal_loss(self.camera_id, signal_msg)
            record = self._fire_alert("signal_loss", "critical",
                                       {"message": signal_msg})
            alerts.append(record.to_dict())

        # 2. Object detection
        detections = self.detector.detect(frame)

        # 3. Object tracking
        tracked = self.tracker.update(detections, frame.shape[:2])

        # 4. Virtual fence check
        intrusions = []
        for det in tracked:
            speed = self.tracker.get_speed(det.track_id) * 0.01  # rough m/s
            bearing = self.tracker.get_bearing(det.track_id)
            events = self.fence.check_intrusion(
                track_id=det.track_id,
                center=det.center,
                object_class=det.class_name,
                speed=speed,
                bearing=bearing,
                confidence=det.confidence,
            )
            for event in events:
                record = self._fire_alert(
                    "fence_intrusion",
                    self.fence.zones.get(event.zone_name,
                                          type('', (), {'severity': 'high'})()).severity
                    if event.zone_name in self.fence.zones else "high",
                    event.to_dict(),
                )
                intrusions.append(event.to_dict())
                alerts.append(record.to_dict())

        # 5. ANPR for vehicles
        plates = []
        vehicles = self.detector.get_vehicles(tracked)
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            # Expand bbox slightly for plate detection
            pad = 20
            crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad),
                         max(0, x1-pad):min(frame.shape[1], x2+pad)]
            if crop.size > 0:
                plate_results = self.anpr.process_frame(crop, self.frame_id)
                for pr in plate_results:
                    self.anpr.add_reading(vehicle.track_id, pr)
                    plates.append(pr.to_dict())

                # Check consensus
                consensus = self.anpr.get_consensus(vehicle.track_id)
                if consensus and consensus.confidence >= 0.7:
                    record = self._fire_alert(
                        "anpr_match", "medium",
                        {
                            "plate_text": consensus.plate_text,
                            "confidence": consensus.confidence,
                            "num_frames": consensus.num_frames,
                            "track_id": vehicle.track_id,
                        }
                    )
                    alerts.append(record.to_dict())

        # 6. Draw annotations
        annotated = self._draw_annotations(frame, tracked, intrusions)

        # 7. Log and compile result
        if self.frame_id % 100 == 0:
            chain_valid = self.hash_chain.verify()[0]
            log.chain(len(self.hash_chain), chain_valid)

        return FrameResult(
            frame_id=self.frame_id,
            timestamp=now,
            detections=[d.to_dict() for d in detections],
            tracks=[d.to_dict() for d in tracked],
            intrusions=intrusions,
            plates=plates,
            alerts=alerts,
            camera_status=self.signal_detector.get_status(),
            annotated_frame=annotated,
            hash_chain_head=self.hash_chain.get_head_hash(),
        )

    def _draw_annotations(self, frame: np.ndarray,
                          detections: List[Detection],
                          intrusions: list) -> np.ndarray:
        """Draw all annotations on the frame."""
        vis = frame.copy()

        # Draw fence zones
        vis = self.fence.draw_zones(vis)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            in_fence = any(
                self.fence.zones[z].contains(det.center)
                for z in self.fence.zones
            )
            color = (0, 0, 255) if in_fence else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"T-{det.track_id:03d} {det.class_name} {det.confidence:.0%}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw trajectory
            traj = self.tracker.get_trajectory(det.track_id)
            if len(traj) > 1:
                pts = np.array(traj[-30:], dtype=np.int32)
                cv2.polylines(vis, [pts], False, (255, 255, 0), 1)

        # Draw intrusion highlights
        for intr in intrusions:
            pt = intr["point"]
            cv2.circle(vis, pt, 15, (0, 0, 255), 3)
            cv2.putText(vis, "INTRUSION!", (pt[0] - 40, pt[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (20, 20, 30), -1)
        status = "ONLINE" if self.signal_detector.get_online_count() > 0 else "OFFLINE"
        hud_text = (f"IBVAP | {self.site_id} {self.camera_id} | "
                    f"Frame {self.frame_id} | "
                    f"Tracks: {len(detections)} | Chain: {len(self.hash_chain)}")
        cv2.putText(vis, hud_text, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

        return vis

    def process_video_file(self, video_path: str, max_frames: int = 300) -> List[FrameResult]:
        """Process a video file frame by frame."""
        cap = cv2.VideoCapture(video_path)
        results = []

        while cap.isOpened() and len(results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.process_frame(frame)
            results.append(result)

        cap.release()
        return results

    def process_webcam(self, max_frames: int = 300) -> List[FrameResult]:
        """Process live webcam feed."""
        cap = cv2.VideoCapture(0)
        results = []

        while cap.isOpened() and len(results) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.process_frame(frame)
            results.append(result)

        cap.release()
        return results

    def get_summary(self) -> dict:
        """Get pipeline summary statistics."""
        chain_stats = self.hash_chain.get_stats()
        return {
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "total_frames": self.frame_id,
            "total_alerts": chain_stats["total_events"],
            "chain_valid": chain_stats["is_valid"],
            "cameras_online": self.signal_detector.get_online_count(),
            "cameras_offline": self.signal_detector.get_offline_count(),
            "fence_zones": len(self.fence.zones),
            "head_hash": chain_stats["head_hash"],
        }
