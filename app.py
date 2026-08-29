"""
IBVAP - Intelligent Border Video Analytics Platform
Main Application Entry Point

Run this to start the complete IBVAP demo.
"""

import cv2
import numpy as np
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue

# Local imports
from src.edge.detector import (
    EdgeDetector,
    ANPREngine,
    SignalLossDetector,
    HashChainVerifier,
    VirtualFence,
    Detection,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class IBVAPConfig:
    """Application configuration"""
    # Camera settings
    camera_source: int = 0  # 0 for webcam, or video file path
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    
    # Detection settings
    detection_model: str = "yolov8n"
    confidence_threshold: float = 0.5
    
    # Virtual fence settings
    enable_virtual_fence: bool = True
    
    # ANPR settings
    enable_anpr: bool = True
    anpr_consensus_threshold: float = 0.6
    
    # Signal loss settings
    signal_loss_timeout: float = 5.0
    
    # Dashboard settings
    enable_dashboard: bool = True
    dashboard_port: int = 8000
    
    # Demo mode
    demo_mode: bool = True  # Use simulated camera feed


# ============================================================
# Simulated Camera Feed
# ============================================================

class SimulatedCamera:
    """
    Simulated camera feed for demo purposes
    Generates realistic border surveillance footage
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.objects = []
        self.time = 0
        
        # Initialize some "objects" to track
        self._init_objects()
    
    def _init_objects(self):
        """Initialize simulated objects"""
        self.objects = [
            {"id": 1, "class": "person", "x": 100, "y": 400, "vx": 2, "vy": 0, "active": True},
            {"id": 2, "class": "vehicle", "x": 800, "y": 500, "vx": -3, "vy": 0, "active": True},
            {"id": 3, "class": "person", "x": 600, "y": 300, "vx": 1, "vy": 1, "active": True},
        ]
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """Read a frame from simulated camera"""
        self.frame_count += 1
        self.time += 1.0 / 30
        
        # Create background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw ground (green area)
        frame[400:, :] = [30, 60, 30]  # Dark green ground
        
        # Draw sky gradient
        for y in range(400):
            blue = int(100 + (y / 400) * 50)
            frame[y, :] = [blue, blue - 20, blue - 40]
        
        # Draw grid (surveillance overlay)
        for x in range(0, self.width, 100):
            cv2.line(frame, (x, 0), (x, self.height), (50, 50, 50), 1)
        for y in range(0, self.height, 100):
            cv2.line(frame, (0, y), (self.width, y), (50, 50, 50), 1)
        
        # Draw virtual fence
        fence_points = np.array([
            [300, 250], [900, 250], [900, 550], [300, 550]
        ], np.int32)
        cv2.polylines(frame, [fence_points], True, (0, 255, 0), 3)
        cv2.putText(frame, "VIRTUAL FENCE", (350, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update and draw objects
        for obj in self.objects:
            if not obj["active"]:
                continue
            
            # Update position
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            
            # Bounce off walls
            if obj["x"] < 50 or obj["x"] > self.width - 50:
                obj["vx"] = -obj["vx"]
            if obj["y"] < 200 or obj["y"] > self.height - 50:
                obj["vy"] = -obj["vy"]
            
            # Draw object
            if obj["class"] == "person":
                # Draw person
                cv2.circle(frame, (int(obj["x"]), int(obj["y"]) - 40), 15, (0, 200, 255), -1)
                cv2.line(frame, (int(obj["x"]), int(obj["y"]) - 25), 
                        (int(obj["x"]), int(obj["y"]) + 20), (0, 200, 255), 3)
                cv2.line(frame, (int(obj["x"]), int(obj["y"])), 
                        (int(obj["x"]) - 15, int(obj["y"]) + 30), (0, 200, 255), 3)
                cv2.line(frame, (int(obj["x"]), int(obj["y"])), 
                        (int(obj["x"]) + 15, int(obj["y"]) + 30), (0, 200, 255), 3)
                
                # Bounding box
                cv2.rectangle(frame, (int(obj["x"]) - 30, int(obj["y"]) - 60),
                             (int(obj["x"]) + 30, int(obj["y"]) + 40), (0, 255, 255), 2)
                cv2.putText(frame, f"Person {85 + int(self.time) % 15}%", 
                           (int(obj["x"]) - 30, int(obj["y"]) - 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            elif obj["class"] == "vehicle":
                # Draw vehicle
                cv2.rectangle(frame, (int(obj["x"]) - 60, int(obj["y"]) - 20),
                             (int(obj["x"]) + 60, int(obj["y"]) + 20), (100, 100, 150), -1)
                cv2.rectangle(frame, (int(obj["x"]) - 40, int(obj["y"]) - 35),
                             (int(obj["x"]) + 40, int(obj["y"]) - 20), (80, 80, 120), -1)
                
                # License plate
                cv2.rectangle(frame, (int(obj["x"]) - 25, int(obj["y"]) + 5),
                             (int(obj["x"]) + 25, int(obj["y"]) + 20), (255, 255, 255), -1)
                plate_text = "BR12AB3456" if int(self.time) % 2 == 0 else "DL01CD7890"
                cv2.putText(frame, plate_text, (int(obj["x"]) - 22, int(obj["y"]) + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Bounding box
                cv2.rectangle(frame, (int(obj["x"]) - 70, int(obj["y"]) - 45),
                             (int(obj["x"]) + 70, int(obj["y"]) + 30), (255, 165, 0), 2)
                cv2.putText(frame, f"Vehicle {90 + int(self.time) % 10}%",
                           (int(obj["x"]) - 70, int(obj["y"]) - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Draw HUD overlay
        self._draw_hud(frame)
        
        return True, frame
    
    def _draw_hud(self, frame: np.ndarray):
        """Draw heads-up display overlay"""
        # Header bar
        cv2.rectangle(frame, (0, 0), (self.width, 50), (20, 20, 30), -1)
        cv2.putText(frame, "IBVAP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (88, 166, 255), 2)
        cv2.putText(frame, "Border Surveillance - LIVE", (150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        # Camera info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (self.width - 250, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Status indicators
        cv2.circle(frame, (self.width - 280, 30), 8, (0, 255, 0), -1)
        cv2.putText(frame, "REC", (self.width - 270, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def release(self):
        """Release camera resources"""
        pass


# ============================================================
# Alert Manager
# ============================================================

class AlertManager:
    """Manage alerts and hash chain"""
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self.prev_hash: str = "0" * 64
        self.alert_queue: queue.Queue = queue.Queue()
    
    def add_alert(self, alert: Dict) -> Dict:
        """Add a new alert with hash chain"""
        # Add hash chain
        alert["prev_hash"] = self.prev_hash
        
        # Calculate new hash
        alert_copy = alert.copy()
        alert_copy.pop("prev_hash", None)
        self.prev_hash = hashlib.sha256(
            json.dumps(alert_copy, sort_keys=True).encode()
        ).hexdigest()
        
        # Store alert
        self.alerts.append(alert)
        
        # Add to queue for real-time updates
        self.alert_queue.put(alert)
        
        return alert
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts[-count:]
    
    def verify_chain(self) -> Tuple[bool, int]:
        """Verify the entire hash chain"""
        return HashChainVerifier.verify_chain(self.alerts)


# ============================================================
# Main IBVAP Application
# ============================================================

class IBVAPApp:
    """
    Main IBVAP Application
    Orchestrates all components for the demo
    """
    
    def __init__(self, config: IBVAPConfig = None):
        self.config = config or IBVAPConfig()
        
        # Initialize components
        print("[IBVAP] Initializing components...")
        
        # Edge detector
        self.detector = EdgeDetector(
            model_name=self.config.detection_model,
            confidence_threshold=self.config.confidence_threshold
        )
        
        # Add virtual fence
        if self.config.enable_virtual_fence:
            fence = VirtualFence(
                fence_id="fence-001",
                zone_name="Zone-1",
                polygon=[(300, 250), (900, 250), (900, 550), (300, 550)],
                severity="high"
            )
            self.detector.add_virtual_fence(fence)
        
        # ANPR engine
        if self.config.enable_anpr:
            self.anpr = ANPREngine()
        
        # Signal loss detector
        self.signal_detector = SignalLossDetector(
            camera_id="CAM-01",
            timeout_seconds=self.config.signal_loss_timeout
        )
        
        # Alert manager
        self.alert_manager = AlertManager()
        
        # Camera
        if self.config.demo_mode:
            self.camera = SimulatedCamera(*self.config.resolution)
        else:
            self.camera = cv2.VideoCapture(self.config.camera_source)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
        print("[IBVAP] Initialization complete!")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        self.frame_count += 1
        timestamp = time.time()
        
        # Run detection
        result = self.detector.process_frame(frame, timestamp)
        
        # Run ANPR on vehicle detections
        if self.config.enable_anpr:
            for det in result["detections"]:
                if det.class_name == "vehicle":
                    # Extract plate region (simplified)
                    plate_text = self.anpr.process_frame(
                        frame, det.track_id, det.bbox
                    )
                    if plate_text:
                        # Create ANPR alert
                        alert = self.alert_manager.add_alert({
                            "event_id": f"anpr-{self.frame_count}",
                            "timestamp": datetime.now().isoformat(),
                            "site_id": "BOP-01",
                            "camera_id": "CAM-01",
                            "event_type": "anpr_match",
                            "object_class": "vehicle",
                            "track_id": f"T-{det.track_id:04d}",
                            "zone": "Checkpoint-1",
                            "confidence": det.confidence,
                            "explanation": f"Vehicle {plate_text} detected at Checkpoint-1",
                            "severity": "medium",
                            "clip_ref": f"s3://ibvap-clips/anpr-{self.frame_count}.mp4"
                        })
        
        # Check signal loss
        signal_alert = self.signal_detector.check_timeout(timestamp)
        if signal_alert:
            alert = self.alert_manager.add_alert(signal_alert)
        
        # Update signal detector
        self.signal_detector.update(timestamp)
        
        # Process fence intrusion alerts
        for alert in result["alerts"]:
            full_alert = self.alert_manager.add_alert({
                "event_id": alert["event_id"],
                "timestamp": alert["timestamp"],
                "site_id": alert["site_id"],
                "camera_id": alert["camera_id"],
                "event_type": alert["event_type"],
                "object_class": alert["object_class"],
                "track_id": alert["track_id"],
                "zone": alert["zone"],
                "confidence": alert["confidence"],
                "explanation": alert["explanation"],
                "severity": alert["severity"],
                "clip_ref": alert["clip_ref"]
            })
        
        return result
    
    def run(self):
        """Run the main application loop"""
        print("\n" + "=" * 60)
        print("IBVAP - Starting Demo")
        print("=" * 60)
        print("\nPress 'q' to quit")
        print("Press 'f' to toggle fullscreen")
        print("Press 'a' to show alert log")
        print("=" * 60 + "\n")
        
        self.is_running = True
        self.start_time = time.time()
        
        # Create window
        cv2.namedWindow("IBVAP - Border Surveillance", cv2.WINDOW_NORMAL)
        
        while self.is_running:
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                print("[IBVAP] Failed to read frame")
                continue
            
            # Process frame
            result = self.process_frame(frame)
            
            # Draw detections on frame
            display_frame = self._draw_detections(frame, result)
            
            # Draw alert panel
            display_frame = self._draw_alert_panel(display_frame)
            
            # Show frame
            cv2.imshow("IBVAP - Border Surveillance", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                current_style = cv2.getWindowProperty("IBVAP - Border Surveillance", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("IBVAP - Border Surveillance", cv2.WND_PROP_FULLSCREEN,
                                     cv2.WINDOW_FULLSCREEN if current_style != cv2.WINDOW_FULLSCREEN else cv2.WINDOW_NORMAL)
            elif key == ord('a'):
                self._show_alert_log()
            
            # Print stats every 100 frames
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"[IBVAP] Frame {self.frame_count} | FPS: {fps:.1f} | "
                      f"Alerts: {len(self.alert_manager.alerts)} | "
                      f"Tracks: {len(self.detector.tracks)}")
        
        # Cleanup
        self.cleanup()
    
    def _draw_detections(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        display = frame.copy()
        
        # Draw track trails
        for track_id, track in self.detector.tracks.items():
            if track.is_active and len(track.detections) > 1:
                # Draw trail
                points = [d.centroid for d in track.detections[-20:]]
                for i in range(1, len(points)):
                    cv2.line(display, points[i-1], points[i], (0, 255, 255), 2)
        
        # Draw stats overlay
        stats = self.detector.get_performance_stats()
        cv2.rectangle(display, (10, 60), (250, 160), (20, 20, 30), -1)
        cv2.putText(display, f"FPS: {stats['avg_fps']:.1f}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Tracks: {stats['total_active_tracks']}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Alerts: {len(self.alert_manager.alerts)}", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Frame: {self.frame_count}", (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display
    
    def _draw_alert_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw alert panel on the right side"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Alert panel background
        panel_width = 350
        cv2.rectangle(display, (w - panel_width, 50), (w, h), (20, 20, 30), -1)
        
        # Panel header
        cv2.putText(display, "ALERTS", (w - panel_width + 10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)
        
        # Draw recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(8)
        y_offset = 110
        
        for alert in recent_alerts:
            # Alert box
            severity = alert.get("severity", "low")
            color = {
                "critical": (156, 39, 176),
                "high": (244, 67, 54),
                "medium": (255, 152, 0),
                "low": (76, 175, 80)
            }.get(severity, (100, 100, 100))
            
            cv2.rectangle(display, (w - panel_width + 10, y_offset),
                         (w - 10, y_offset + 60), color, 2)
            
            # Alert text
            cv2.putText(display, alert.get("event_type", "unknown").upper(),
                       (w - panel_width + 20, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, alert.get("zone", ""),
                       (w - panel_width + 20, y_offset + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Time
            time_str = alert.get("timestamp", "")[:19]
            cv2.putText(display, time_str,
                       (w - panel_width + 20, y_offset + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            
            y_offset += 70
        
        # Hash chain status
        is_valid, tampered_idx = self.alert_manager.verify_chain()
        chain_status = "VALID" if is_valid else "TAMPERED"
        chain_color = (0, 255, 0) if is_valid else (0, 0, 255)
        
        cv2.rectangle(display, (w - panel_width + 10, h - 50),
                     (w - 10, h - 10), (30, 30, 40), -1)
        cv2.putText(display, f"HASH CHAIN: {chain_status}",
                   (w - panel_width + 20, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, chain_color, 1)
        
        return display
    
    def _show_alert_log(self):
        """Show full alert log in console"""
        print("\n" + "=" * 60)
        print("ALERT LOG")
        print("=" * 60)
        
        for i, alert in enumerate(self.alert_manager.alerts):
            print(f"\n[{i+1}] {alert.get('event_type', 'unknown').upper()}")
            print(f"    Time: {alert.get('timestamp', 'N/A')}")
            print(f"    Zone: {alert.get('zone', 'N/A')}")
            print(f"    Severity: {alert.get('severity', 'N/A')}")
            print(f"    Explanation: {alert.get('explanation', 'N/A')}")
        
        print("\n" + "=" * 60)
        is_valid, tampered_idx = self.alert_manager.verify_chain()
        print(f"Hash Chain: {'VALID' if is_valid else f'TAMPERED at index {tampered_idx}'}")
        print("=" * 60 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[IBVAP] Shutting down...")
        self.is_running = False
        self.camera.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n[IBVAP] Final Statistics:")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Total time: {elapsed:.1f} seconds")
            print(f"  Average FPS: {self.frame_count / elapsed:.1f}")
            print(f"  Total alerts: {len(self.alert_manager.alerts)}")
            
            # Verify hash chain
            is_valid, _ = self.alert_manager.verify_chain()
            print(f"  Hash chain: {'VALID' if is_valid else 'TAMPERED'}")


# ============================================================
# Entry Point
# ============================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IBVAP - Intelligent Border Video Analytics Platform")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (0 for webcam)")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--no-demo", action="store_true", help="Disable demo mode (use real camera)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--model", type=str, default="yolov8n", help="Detection model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Create config
    config = IBVAPConfig(
        camera_source=args.video if args.video else args.camera,
        resolution=(args.width, args.height),
        detection_model=args.model,
        confidence_threshold=args.confidence,
        demo_mode=not args.no_demo
    )
    
    # Create and run application
    app = IBVAPApp(config)
    app.run()


if __name__ == "__main__":
    main()
