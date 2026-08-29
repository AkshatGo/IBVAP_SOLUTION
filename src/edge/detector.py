"""
IBVAP Edge Detection Module
YOLOv8-nano + ByteTrack for real-time border surveillance
Runs on Jetson Orin Nano or laptop for demo
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import hashlib
import json
from collections import defaultdict


@dataclass
class Detection:
    """Single detection result"""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[int, int]
    timestamp: float
    frame_idx: int


@dataclass
class Track:
    """Track history for a single object"""
    track_id: int
    class_name: str
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen
    
    @property
    def velocity(self) -> Tuple[float, float]:
        """Average velocity in pixels/second"""
        if len(self.detections) < 2:
            return (0.0, 0.0)
        dx = self.detections[-1].centroid[0] - self.detections[0].centroid[0]
        dy = self.detections[-1].centroid[1] - self.detections[0].centroid[1]
        dt = self.detections[-1].timestamp - self.detections[0].timestamp
        if dt < 1e-6:
            return (0.0, 0.0)
        return (dx / dt, dy / dt)


@dataclass
class VirtualFence:
    """Virtual fence polygon for intrusion detection"""
    fence_id: str
    zone_name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) vertices
    severity: str = "high"
    is_active: bool = True


class EdgeDetector:
    """
    Edge detection module using YOLOv8 + ByteTrack
    Designed for real-time border surveillance
    """
    
    def __init__(self, model_name: str = "yolov8n", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.tracks: Dict[int, Track] = {}
        self.virtual_fences: List[VirtualFence] = []
        self.alert_history: List[Dict] = []
        self.prev_hash: str = "0" * 64  # Genesis hash for tamper-evident log
        
        # Performance metrics
        self.frame_count: int = 0
        self.total_detection_time: float = 0.0
        
        print(f"[EdgeDetector] Initialized with {model_name}")
        print(f"[EdgeDetector] Confidence threshold: {confidence_threshold}")
    
    def add_virtual_fence(self, fence: VirtualFence) -> None:
        """Add a virtual fence for intrusion detection"""
        self.virtual_fences.append(fence)
        print(f"[EdgeDetector] Added virtual fence: {fence.zone_name}")
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """
        Process a single frame and return detections + alerts
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            timestamp: Unix timestamp
            
        Returns:
            Dictionary with detections, alerts, and metadata
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Step 1: Run YOLOv8 detection
        raw_detections = self._run_detection(frame)
        
        # Step 2: Run ByteTrack for tracking
        tracked_objects = self._run_tracking(raw_detections, timestamp)
        
        # Step 3: Check virtual fence intrusions
        alerts = self._check_virtual_fences(tracked_objects, timestamp)
        
        # Step 4: Update track history
        self._update_tracks(tracked_objects, timestamp)
        
        # Calculate performance
        detection_time = time.time() - start_time
        self.total_detection_time += detection_time
        
        return {
            "frame_idx": self.frame_count,
            "timestamp": timestamp,
            "detections": tracked_objects,
            "alerts": alerts,
            "active_tracks": len([t for t in self.tracks.values() if t.is_active]),
            "detection_time_ms": detection_time * 1000,
            "fps": 1.0 / max(detection_time, 1e-6)
        }
    
    def _run_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 detection on frame
        In production, this calls the actual YOLOv8 model
        """
        # Simulated detection for demo purposes
        # In real implementation, use:
        # from ultralytics import YOLO
        # model = YOLO('yolov8n.pt')
        # results = model(frame)
        
        detections = []
        
        # Simulate detecting people and vehicles
        h, w = frame.shape[:2] if frame is not None else (480, 640)
        
        # Demo: generate some sample detections
        if self.frame_count % 30 == 0:  # Every 30 frames, simulate a detection
            detections.append({
                "class_name": "person",
                "confidence": 0.87,
                "bbox": (100, 150, 200, 350),
                "centroid": (150, 250)
            })
        
        if self.frame_count % 45 == 0:  # Every 45 frames, simulate a vehicle
            detections.append({
                "class_name": "vehicle",
                "confidence": 0.92,
                "bbox": (300, 200, 500, 350),
                "centroid": (400, 275)
            })
        
        return detections
    
    def _run_tracking(self, detections: List[Dict], timestamp: float) -> List[Detection]:
        """
        Run ByteTrack for object tracking
        Assigns persistent track IDs across frames
        """
        tracked = []
        
        for det in detections:
            # In production, use ByteTrack:
            # from byte_tracker import ByteTracker
            # tracker = ByteTracker()
            # tracked_objects = tracker.update(detections)
            
            # Assign track ID (simplified for demo)
            track_id = self._assign_track_id(det["centroid"], det["class_name"])
            
            detection = Detection(
                track_id=track_id,
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                centroid=det["centroid"],
                timestamp=timestamp,
                frame_idx=self.frame_count
            )
            tracked.append(detection)
        
        return tracked
    
    def _assign_track_id(self, centroid: Tuple[int, int], class_name: str) -> int:
        """Assign track ID based on proximity to existing tracks"""
        # Simple nearest-neighbor tracking for demo
        # In production, use ByteTrack's Kalman filter
        
        best_id = len(self.tracks) + 1
        min_dist = float('inf')
        
        for track_id, track in self.tracks.items():
            if track.is_active and track.class_name == class_name:
                last_det = track.detections[-1] if track.detections else None
                if last_det:
                    dist = np.sqrt(
                        (centroid[0] - last_det.centroid[0]) ** 2 +
                        (centroid[1] - last_det.centroid[1]) ** 2
                    )
                    if dist < min_dist and dist < 100:  # 100px threshold
                        min_dist = dist
                        best_id = track_id
        
        return best_id
    
    def _update_tracks(self, detections: List[Detection], timestamp: float) -> None:
        """Update track history with new detections"""
        for det in detections:
            if det.track_id not in self.tracks:
                self.tracks[det.track_id] = Track(
                    track_id=det.track_id,
                    class_name=det.class_name,
                    first_seen=timestamp
                )
            
            self.tracks[det.track_id].detections.append(det)
            self.tracks[det.track_id].last_seen = timestamp
            self.tracks[det.track_id].is_active = True
    
    def _check_virtual_fences(self, detections: List[Detection], timestamp: float) -> List[Dict]:
        """Check if any detected objects have crossed virtual fences"""
        alerts = []
        
        for det in detections:
            for fence in self.virtual_fences:
                if not fence.is_active:
                    continue
                
                # Check if centroid is inside polygon
                if self._point_in_polygon(det.centroid, fence.polygon):
                    # Calculate explanation
                    explanation = self._generate_explanation(det, fence)
                    
                    # Create alert with hash chain
                    alert = self._create_alert(
                        event_type="fence_intrusion",
                        detection=det,
                        fence=fence,
                        explanation=explanation,
                        timestamp=timestamp
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _generate_explanation(self, detection: Detection, fence: VirtualFence) -> str:
        """Generate human-readable explanation for alert"""
        # Calculate bearing
        bearing = self._calculate_bearing(detection)
        
        # Calculate speed
        if detection.track_id in self.tracks:
            track = self.tracks[detection.track_id]
            velocity = track.velocity
            speed_mps = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) * 0.01  # Convert to m/s
        else:
            speed_mps = 0.0
        
        explanation = (
            f"Track T-{detection.track_id:04d} crossed virtual fence "
            f"{fence.zone_name} at {speed_mps:.1f} m/s, bearing {bearing}. "
            f"Class: {detection.class_name}, confidence: {detection.confidence:.2f}"
        )
        
        return explanation
    
    def _calculate_bearing(self, detection: Detection) -> str:
        """Calculate bearing from track history"""
        if detection.track_id not in self.tracks:
            return "N/A"
        
        track = self.tracks[detection.track_id]
        if len(track.detections) < 2:
            return "N/A"
        
        # Calculate direction from last two detections
        prev = track.detections[-2].centroid
        curr = track.detections[-1].centroid
        
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        
        # Convert to compass bearing
        angle = np.degrees(np.arctan2(dx, -dy))  # -dy because y increases downward
        bearing = (angle + 360) % 360
        
        # Convert to cardinal direction
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = round(bearing / 45) % 8
        return directions[idx]
    
    def _create_alert(self, event_type: str, detection: Detection, 
                     fence: VirtualFence, explanation: str, timestamp: float) -> Dict:
        """Create tamper-evident alert with hash chain"""
        event_id = hashlib.sha256(
            f"{timestamp}{detection.track_id}{fence.fence_id}".encode()
        ).hexdigest()[:16]
        
        alert = {
            "event_id": event_id,
            "prev_hash": self.prev_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "site_id": "BOP-01",
            "camera_id": "CAM-01",
            "event_type": event_type,
            "object_class": detection.class_name,
            "track_id": f"T-{detection.track_id:04d}",
            "zone": fence.zone_name,
            "confidence": detection.confidence,
            "explanation": explanation,
            "severity": fence.severity,
            "clip_ref": f"s3://ibvap-clips/{event_id}.mp4"
        }
        
        # Update hash chain
        self.prev_hash = hashlib.sha256(
            json.dumps(alert, sort_keys=True).encode()
        ).hexdigest()
        
        # Store in history
        self.alert_history.append(alert)
        
        return alert
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_detection_time = (
            self.total_detection_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            "total_frames": self.frame_count,
            "total_active_tracks": len([t for t in self.tracks.values() if t.is_active]),
            "total_alerts": len(self.alert_history),
            "avg_detection_time_ms": avg_detection_time * 1000,
            "avg_fps": 1.0 / max(avg_detection_time, 1e-6)
        }


class ANPREngine:
    """
    ANPR with multi-frame OCR consensus voting
    More robust than single-frame OCR
    """
    
    def __init__(self, ocr_model: str = "paddleocr"):
        self.ocr_model = ocr_model
        self.frame_results: Dict[int, List[str]] = defaultdict(list)
        self.consensus_threshold: float = 0.6  # 60% agreement needed
        
        print(f"[ANPREngine] Initialized with {ocr_model}")
    
    def process_frame(self, frame: np.ndarray, track_id: int, 
                     bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Process a single frame for ANPR
        Returns consensus plate number if available
        """
        # Extract license plate region
        plate_region = self._extract_plate_region(frame, bbox)
        
        # Run OCR
        plate_text = self._run_ocr(plate_region)
        
        if plate_text:
            self.frame_results[track_id].append(plate_text)
        
        # Check for consensus
        return self._get_consensus(track_id)
    
    def _extract_plate_region(self, frame: np.ndarray, 
                             bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract license plate region from frame"""
        x1, y1, x2, y2 = bbox
        
        # In production, use license plate detection model
        # For demo, extract lower portion of bbox (where plate usually is)
        plate_height = int((y2 - y1) * 0.3)
        plate_bbox = (x1, y2 - plate_height, x2, y2)
        
        return frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
    
    def _run_ocr(self, plate_region: np.ndarray) -> Optional[str]:
        """Run OCR on plate region"""
        # In production, use PaddleOCR or EasyOCR
        # results = ocr.ocr(plate_region)
        
        # Simulated OCR for demo
        if plate_region is not None and plate_region.size > 0:
            # Simulate detecting a plate
            import random
            if random.random() > 0.3:  # 70% success rate
                plates = ["BR12AB3456", "DL01CD7890", "MH02EF1234"]
                return random.choice(plates)
        
        return None
    
    def _get_consensus(self, track_id: int) -> Optional[str]:
        """Get consensus plate from multiple frames"""
        results = self.frame_results[track_id]
        
        if len(results) < 3:  # Need at least 3 frames
            return None
        
        # Count occurrences
        from collections import Counter
        counter = Counter(results)
        
        # Find majority
        most_common, count = counter.most_common(1)[0]
        agreement = count / len(results)
        
        if agreement >= self.consensus_threshold:
            return most_common
        
        return None
    
    def get_stats(self) -> Dict:
        """Get ANPR statistics"""
        total_frames = sum(len(v) for v in self.frame_results.values())
        consensus_count = sum(
            1 for track_id in self.frame_results 
            if self._get_consensus(track_id) is not None
        )
        
        return {
            "total_frames_processed": total_frames,
            "tracks_with_consensus": consensus_count,
            "consensus_rate": consensus_count / max(len(self.frame_results), 1)
        }


class SignalLossDetector:
    """
    Detects camera signal loss and generates high-severity alerts
    Key innovation: signal loss is itself an alert
    """
    
    def __init__(self, camera_id: str, timeout_seconds: float = 5.0):
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.last_frame_time: float = 0.0
        self.is_online: bool = True
        self.prev_hash: str = "0" * 64
        
        print(f"[SignalLossDetector] Initialized for {camera_id}")
    
    def update(self, timestamp: float) -> Optional[Dict]:
        """
        Update with new frame timestamp
        Returns alert if signal loss detected
        """
        self.last_frame_time = timestamp
        self.is_online = True
        return None
    
    def check_timeout(self, current_time: float) -> Optional[Dict]:
        """Check if camera has timed out"""
        if self.is_online and (current_time - self.last_frame_time) > self.timeout_seconds:
            self.is_online = False
            return self._create_signal_loss_alert(current_time)
        
        return None
    
    def _create_signal_loss_alert(self, timestamp: float) -> Dict:
        """Create high-severity signal loss alert"""
        event_id = hashlib.sha256(
            f"signal_loss_{self.camera_id}_{timestamp}".encode()
        ).hexdigest()[:16]
        
        alert = {
            "event_id": event_id,
            "prev_hash": self.prev_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "site_id": "BOP-01",
            "camera_id": self.camera_id,
            "event_type": "signal_loss",
            "object_class": "none",
            "track_id": "N/A",
            "zone": "N/A",
            "confidence": 1.0,
            "explanation": f"Camera {self.camera_id} signal lost for {self.timeout_seconds} seconds",
            "severity": "critical",
            "clip_ref": "N/A"
        }
        
        # Update hash chain
        self.prev_hash = hashlib.sha256(
            json.dumps(alert, sort_keys=True).encode()
        ).hexdigest()
        
        return alert


class HashChainVerifier:
    """
    Verify tamper-evident hash chain
    Detects any modifications to event history
    """
    
    @staticmethod
    def verify_chain(events: List[Dict]) -> Tuple[bool, Optional[int]]:
        """
        Verify the integrity of the event hash chain
        
        Returns:
            Tuple of (is_valid, tampered_event_index)
        """
        if not events:
            return True, None
        
        for i in range(1, len(events)):
            # Recalculate hash of previous event
            prev_event = events[i - 1].copy()
            prev_event.pop("prev_hash", None)
            
            calculated_hash = hashlib.sha256(
                json.dumps(prev_event, sort_keys=True).encode()
            ).hexdigest()
            
            # Compare with stored hash
            if calculated_hash != events[i]["prev_hash"]:
                return False, i
        
        return True, None


# Demo function
def demo():
    """Run a quick demo of the edge detection module"""
    print("=" * 60)
    print("IBVAP Edge Detection Demo")
    print("=" * 60)
    
    # Initialize detector
    detector = EdgeDetector(model_name="yolov8n", confidence_threshold=0.5)
    
    # Add virtual fence
    fence = VirtualFence(
        fence_id="fence-001",
        zone_name="Zone-3",
        polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
        severity="high"
    )
    detector.add_virtual_fence(fence)
    
    # Initialize ANPR
    anpr = ANPREngine(ocr_model="paddleocr")
    
    # Initialize signal loss detector
    signal_detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=5.0)
    
    # Process 100 frames
    print("\nProcessing 100 frames...")
    for i in range(100):
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time() + i * 0.033  # 30 FPS
        
        # Process frame
        result = detector.process_frame(frame, timestamp)
        
        # Update signal detector
        signal_detector.update(timestamp)
        
        # Print progress
        if i % 20 == 0:
            print(f"Frame {i}: {result['active_tracks']} active tracks, "
                  f"{len(result['alerts'])} alerts, "
                  f"{result['fps']:.1f} FPS")
    
    # Get statistics
    stats = detector.get_performance_stats()
    print("\n" + "=" * 60)
    print("Performance Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Active tracks: {stats['total_active_tracks']}")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Avg detection time: {stats['avg_detection_time_ms']:.2f} ms")
    print(f"  Avg FPS: {stats['avg_fps']:.1f}")
    
    # Verify hash chain
    if detector.alert_history:
        verifier = HashChainVerifier()
        is_valid, tampered_idx = verifier.verify_chain(detector.alert_history)
        print(f"\nHash chain integrity: {'VALID' if is_valid else 'TAMPERED'}")
        if tampered_idx is not None:
            print(f"  Tampered at event index: {tampered_idx}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
