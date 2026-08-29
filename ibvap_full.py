"""
IBVAP - Full Working Prototype
Real YOLOv8 detection + ANPR + Surveillance
"""

import cv2
import numpy as np
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Detection
    model_name: str = "yolov8n"
    confidence: float = 0.45
    nms_threshold: float = 0.45
    
    # Classes we detect (COCO)
    person_class: int = 0
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck
    
    # Virtual fence
    fence_points: List[Tuple[int, int]] = field(default_factory=lambda: [
        (250, 200), (950, 200), (950, 550), (250, 550)
    ])
    
    # Signal loss
    signal_loss_timeout: float = 5.0
    
    # Display
    width: int = 1280
    height: int = 720


# ============================================================
# Detection Engine (uses real YOLOv8)
# ============================================================

class DetectionEngine:
    """Real object detection using YOLOv8"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{self.config.model_name}.pt")
            print(f"✓ Loaded {self.config.model_name}")
        except Exception as e:
            print(f"✗ Could not load YOLO: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on frame"""
        if self.model is None:
            return []
        
        results = self.model(frame, conf=self.config.confidence, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Classify
                    obj_class = "unknown"
                    if cls == self.config.person_class:
                        obj_class = "person"
                    elif cls in self.config.vehicle_classes:
                        obj_class = "vehicle"
                    
                    if obj_class != "unknown":
                        detections.append({
                            "class": obj_class,
                            "class_id": cls,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2),
                            "centroid": ((x1 + x2) // 2, (y1 + y2) // 2)
                        })
        
        return detections


# ============================================================
# Tracking
# ============================================================

class SimpleTracker:
    """Simple IoU-based tracker"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched = set()
        matched_tracks = set()
        
        for det in detections:
            best_iou = 0
            best_id = None
            
            for track_id, track in self.tracks.items():
                if track["age"] > self.max_age:
                    continue
                
                iou = self._compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id
            
            if best_iou > self.iou_threshold and best_id is not None:
                # Update existing track
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["centroid"] = det["centroid"]
                self.tracks[best_id]["class"] = det["class"]
                self.tracks[best_id]["confidence"] = det["confidence"]
                self.tracks[best_id]["age"] = 0
                self.tracks[best_id]["history"].append(det["centroid"])
                det["track_id"] = best_id
                matched.add(best_id)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "bbox": det["bbox"],
                    "centroid": det["centroid"],
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "age": 0,
                    "history": [det["centroid"]],
                    "first_seen": self.frame_count
                }
                det["track_id"] = track_id
                matched.add(track_id)
            
            matched_tracks.add(det["track_id"])
        
        # Age unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched:
                self.tracks[track_id]["age"] += 1
        
        return detections
    
    def _compute_iou(self, bbox1, bbox2) -> float:
        """Compute IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def get_active_tracks(self) -> Dict:
        """Get currently active tracks"""
        return {k: v for k, v in self.tracks.items() if v["age"] <= self.max_age}


# ============================================================
# ANPR (simplified - plate detection from vehicle crops)
# ============================================================

class ANPRSystem:
    """Simplified ANPR system"""
    
    def __init__(self):
        self.plates: Dict[int, str] = {}
        self.frame_buffer: Dict[int, List[str]] = {}
    
    def process_vehicle(self, track_id: int, frame: np.ndarray, bbox: Tuple) -> Optional[str]:
        """Process a vehicle frame for plate detection"""
        x1, y1, x2, y2 = bbox
        
        # Extract vehicle region
        vehicle_crop = frame[max(0, y1):min(frame.shape[0], y2), 
                           max(0, x1):min(frame.shape[1], x2)]
        
        if vehicle_crop.size == 0:
            return None
        
        # Simple plate detection heuristic (bottom center of vehicle)
        h, w = vehicle_crop.shape[:2]
        plate_region = vehicle_crop[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
        
        if plate_region.size == 0:
            return None
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if it looks like a plate (white rectangle with text)
        white_ratio = np.sum(thresh > 128) / max(thresh.size, 1)
        
        if 0.3 < white_ratio < 0.8:
            # Simulate plate reading (in production, use OCR)
            plate_text = self._simulate_ocr(track_id)
            
            if track_id not in self.frame_buffer:
                self.frame_buffer[track_id] = []
            
            self.frame_buffer[track_id].append(plate_text)
            
            # Consensus voting
            if len(self.frame_buffer[track_id]) >= 3:
                from collections import Counter
                counter = Counter(self.frame_buffer[track_id])
                plate_text = counter.most_common(1)[0][0]
                self.plates[track_id] = plate_text
                return plate_text
        
        return None
    
    def _simulate_ocr(self, track_id: int) -> str:
        """Simulate OCR (replace with PaddleOCR in production)"""
        plates = ["BR12AB3456", "DL01CD7890", "MH02EF1234", "KA03GH5678"]
        return plates[track_id % len(plates)]


# ============================================================
# Alert System with Hash Chain
# ============================================================

class AlertSystem:
    """Alert generation with tamper-evident hash chain"""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.prev_hash: str = "0" * 64
        self.last_alert_time: Dict[str, float] = {}
        self.cooldown: float = 2.0  # seconds between same alerts
    
    def create_alert(self, event_type: str, explanation: str, severity: str,
                    track_id: int = 0, zone: str = "Zone-1") -> Optional[Dict]:
        """Create alert with cooldown"""
        now = time.time()
        
        # Cooldown check
        key = f"{event_type}_{track_id}"
        if key in self.last_alert_time:
            if now - self.last_alert_time[key] < self.cooldown:
                return None
        
        self.last_alert_time[key] = now
        
        alert = {
            "event_id": f"e{len(self.alerts) + 1:04d}",
            "prev_hash": self.prev_hash,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "track_id": f"T-{track_id:04d}",
            "zone": zone,
            "severity": severity,
            "explanation": explanation
        }
        
        # Hash chain
        alert_copy = alert.copy()
        alert_copy.pop("prev_hash", None)
        self.prev_hash = hashlib.sha256(
            json.dumps(alert_copy, sort_keys=True).encode()
        ).hexdigest()
        
        self.alerts.append(alert)
        return alert
    
    def verify_chain(self) -> bool:
        """Verify hash chain integrity"""
        for i in range(1, len(self.alerts)):
            prev = self.alerts[i - 1].copy()
            prev.pop("prev_hash", None)
            h = hashlib.sha256(json.dumps(prev, sort_keys=True).encode()).hexdigest()
            if h != self.alerts[i]["prev_hash"]:
                return False
        return True
    
    def get_recent(self, n: int = 8) -> List[Dict]:
        return self.alerts[-n:]


# ============================================================
# Signal Loss Detector
# ============================================================

class SignalLossDetector:
    def __init__(self, cameras: List[str], timeout: float = 5.0):
        self.cameras = {cam: {"online": True, "last_frame": time.time()} for cam in cameras}
        self.timeout = timeout
    
    def update(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id]["last_frame"] = time.time()
            self.cameras[camera_id]["online"] = True
    
    def check(self) -> List[Dict]:
        now = time.time()
        alerts = []
        for cam, info in self.cameras.items():
            if info["online"] and (now - info["last_frame"]) > self.timeout:
                info["online"] = False
                alerts.append({"camera": cam, "type": "signal_loss"})
        return alerts
    
    def simulate_loss(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id]["last_frame"] = 0
    
    def restore(self, camera_id: str):
        if camera_id in self.cameras:
            self.cameras[camera_id]["online"] = True
            self.cameras[camera_id]["last_frame"] = time.time()
    
    def get_status(self) -> Dict[str, bool]:
        return {cam: info["online"] for cam, info in self.cameras.items()}


# ============================================================
# Main IBVAP Application
# ============================================================

class IBVAP:
    """Full IBVAP application"""
    
    def __init__(self):
        self.config = Config()
        self.detection = DetectionEngine(self.config)
        self.tracker = SimpleTracker()
        self.anpr = ANPRSystem()
        self.alerts = AlertSystem()
        self.signal = SignalLossDetector(["CAM-01", "CAM-02", "CAM-03"])
        self.fence = np.array(self.config.fence_points, np.int32)
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame"""
        self.frame_count += 1
        now = time.time()
        
        if self.start_time:
            elapsed = now - self.start_time
            self.fps = self.frame_count / max(elapsed, 0.001)
        
        display = frame.copy()
        events = []
        
        # 1. Detect objects
        detections = self.detection.detect(frame)
        
        # 2. Track
        tracked = self.tracker.update(detections)
        
        # 3. Check fence + generate alerts
        for det in tracked:
            cx, cy = det["centroid"]
            in_fence = cv2.pointPolygonTest(self.fence, (float(cx), float(cy)), False) >= 0
            
            # Draw detection
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 0, 255) if in_fence else (0, 255, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class'].title()} {det['confidence']:.0%}"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if det["class"] == "person" and in_fence:
                alert = self.alerts.create_alert(
                    "fence_intrusion",
                    f"Person T-{det.get('track_id', 0):04d} crossed Zone-1",
                    "high",
                    det.get("track_id", 0)
                )
                if alert:
                    events.append(alert)
                
                # Draw alert indicator
                cv2.putText(display, "⚠ ALERT", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ANPR for vehicles
            if det["class"] == "vehicle":
                plate = self.anpr.process_vehicle(det.get("track_id", 0), frame, det["bbox"])
                if plate:
                    cv2.putText(display, plate, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw track trail
            track_id = det.get("track_id")
            if track_id and track_id in self.tracker.tracks:
                history = self.tracker.tracks[track_id]["history"][-15:]
                for i in range(1, len(history)):
                    cv2.line(display, history[i-1], history[i], (255, 255, 0), 2)
        
        # 4. Check signal loss
        signal_alerts = self.signal.check()
        for sa in signal_alerts:
            alert = self.alerts.create_alert(
                "signal_loss",
                f"Camera {sa['camera']} signal lost",
                "critical"
            )
            if alert:
                events.append(alert)
        
        # 5. Draw virtual fence
        cv2.polylines(display, [self.fence], True, (0, 255, 0), 3)
        cv2.putText(display, "VIRTUAL FENCE", (350, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 6. Draw HUD
        display = self._draw_hud(display)
        
        return display, events
    
    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw heads-up display"""
        h, w = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 25), -1)
        cv2.putText(frame, "IBVAP", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (88, 166, 255), 3)
        cv2.putText(frame, "Intelligent Border Video Analytics", (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 130, 140), 1)
        
        # Recording
        rec = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 0, 80)
        cv2.circle(frame, (w - 80, 30), 8, rec, -1)
        cv2.putText(frame, "LIVE", (w - 65, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats
        cv2.rectangle(frame, (10, 60), (220, 180), (20, 20, 30), -1)
        stats = [
            f"FPS: {self.fps:.0f}",
            f"Tracks: {len(self.tracker.get_active_tracks())}",
            f"Alerts: {len(self.alerts.alerts)}",
            f"Frame: {self.frame_count}",
        ]
        for i, s in enumerate(stats):
            cv2.putText(frame, s, (20, 85 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Camera status
        cam_status = self.signal.get_status()
        cv2.rectangle(frame, (10, 190), (220, 190 + 25 * len(cam_status)), (20, 20, 30), -1)
        cv2.putText(frame, "CAMERAS", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 166, 255), 1)
        for i, (cam, online) in enumerate(cam_status.items()):
            y = 230 + i * 22
            c = (0, 200, 0) if online else (0, 0, 255)
            cv2.circle(frame, (25, y), 5, c, -1)
            cv2.putText(frame, cam, (35, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        # Alert panel
        panel_x = w - 350
        cv2.rectangle(frame, (panel_x, 55), (w, h), (15, 15, 25), -1)
        cv2.line(frame, (panel_x, 55), (panel_x, h), (88, 166, 255), 2)
        cv2.putText(frame, "ALERTS", (panel_x + 15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)
        
        for i, alert in enumerate(reversed(self.alerts.get_recent(7))):
            y = 110 + i * 65
            sev = alert["severity"]
            c = {"critical": (156, 39, 176), "high": (244, 67, 54), "medium": (255, 152, 0)}.get(sev, (100, 100, 100))
            cv2.rectangle(frame, (panel_x + 10, y), (w - 10, y + 55), c, 2)
            cv2.putText(frame, alert["event_type"].upper(), (panel_x + 20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, alert["explanation"][:45], (panel_x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        # Hash chain
        valid = self.alerts.verify_chain()
        cv2.putText(frame, f"CHAIN: {'VALID' if valid else 'TAMPERED'}", (panel_x + 15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0) if valid else (0, 0, 255), 1)
        
        return frame
    
    def run(self):
        """Run the application"""
        print("\n" + "=" * 60)
        print("  IBVAP - Intelligent Border Video Analytics Platform")
        print("  Full Working Prototype with Real YOLOv8 Detection")
        print("=" * 60)
        print("\n  q = Quit | f = Fullscreen | a = Alert Log | s = Signal Loss")
        print("=" * 60 + "\n")
        
        self.start_time = time.time()
        
        # Try to open camera, fall back to synthetic
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  No webcam found. Using synthetic demo mode.")
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            print("  Camera opened successfully!")
        
        cv2.namedWindow("IBVAP", cv2.WINDOW_NORMAL)
        
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    continue
            else:
                # Synthetic frame
                frame = self._make_synthetic_frame()
            
            display, events = self.process_frame(frame)
            cv2.imshow("IBVAP", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                cv2.setWindowProperty("IBVAP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('a'):
                self._show_alert_log()
            elif key == ord('s'):
                self.signal.simulate_loss("CAM-03")
        
        self._cleanup(cap)
    
    def _make_synthetic_frame(self) -> np.ndarray:
        """Create synthetic frame for demo without camera"""
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        
        # Sky
        for y in range(300):
            b = int(80 + (y / 300) * 50)
            frame[y, :] = [b, b - 10, b - 25]
        
        # Ground
        frame[300:, :] = [35, 55, 35]
        
        # Background objects
        t = time.time()
        
        # Simulated person moving
        px = int(400 + 200 * np.sin(t * 0.5))
        py = int(380 + 50 * np.cos(t * 0.3))
        cv2.rectangle(frame, (px - 20, py - 50), (px + 20, py + 10), (0, 200, 255), -1)
        cv2.circle(frame, (px, py - 55), 12, (0, 200, 255), -1)
        
        # Simulated vehicle
        vx = int(800 - 300 * ((t * 0.3) % 1))
        vy = 450
        cv2.rectangle(frame, (vx - 50, vy - 20), (vx + 50, vy + 20), (100, 100, 150), -1)
        cv2.rectangle(frame, (vx - 35, vy - 32), (vx + 35, vy - 20), (80, 80, 120), -1)
        
        # Border fence
        cv2.line(frame, (50, 300), (self.config.width - 50, 300), (100, 100, 100), 3)
        
        return frame
    
    def _show_alert_log(self):
        print("\n" + "=" * 60)
        print("  ALERT LOG")
        print("=" * 60)
        for a in self.alerts.alerts:
            print(f"  [{a['event_type']}] {a['explanation']} ({a['severity']})")
        print(f"\n  Hash Chain: {'VALID' if self.alerts.verify_chain() else 'TAMPERED'}")
        print(f"  Total Alerts: {len(self.alerts.alerts)}")
        print("=" * 60 + "\n")
    
    def _cleanup(self, cap):
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - self.start_time
        print(f"\n  Session: {self.frame_count} frames, {elapsed:.1f}s, {self.fps:.1f} FPS")
        print(f"  Alerts: {len(self.alerts.alerts)}, Chain: {'VALID' if self.alerts.verify_chain() else 'TAMPERED'}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app = IBVAP()
    app.run()
