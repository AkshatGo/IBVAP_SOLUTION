"""
IBVAP Complete Demo
Full working prototype for SIH 2026 pitch presentation
"""

import cv2
import numpy as np
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys

# ============================================================
# Configuration
# ============================================================

@dataclass
class DemoConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    demo_duration: int = 60  # seconds


# ============================================================
# Simulated Border Scene Generator
# ============================================================

class BorderSceneGenerator:
    """Generate realistic border surveillance scenes for demo"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.time = 0
        
        # Objects in the scene
        self.objects = []
        self._init_scene()
    
    def _init_scene(self):
        """Initialize scene objects"""
        self.objects = [
            {"id": 1, "class": "person", "x": 100, "y": 400, "vx": 2.5, "vy": 0.5, "active": True, "in_fence": False},
            {"id": 2, "class": "vehicle", "x": 900, "y": 500, "vx": -4, "vy": 0, "active": True, "plate": "BR12AB3456"},
            {"id": 3, "class": "person", "x": 700, "y": 350, "vx": 1, "vy": 1.5, "active": True, "in_fence": False},
            {"id": 4, "class": "person", "x": 200, "y": 200, "vx": 0, "vy": 0, "active": True, "in_fence": False},
            {"id": 5, "class": "vehicle", "x": 50, "y": 550, "vx": 3, "vy": -0.5, "active": True, "plate": "DL01CD7890"},
        ]
        
        # Virtual fence vertices
        self.fence_points = np.array([
            [300, 250], [950, 250], [950, 600], [300, 600]
        ], np.int32)
        
        # Camera status
        self.cameras_online = {"CAM-01": True, "CAM-02": True, "CAM-03": True}
        self.signal_loss_timer = 0
    
    def generate_frame(self) -> Tuple[bool, np.ndarray]:
        """Generate a single frame of the border scene"""
        self.frame_count += 1
        self.time += 1.0 / self.fps
        
        # Create base frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(300):
            blue = int(80 + (y / 300) * 60)
            frame[y, :] = [blue, blue - 10, blue - 30]
        
        # Ground
        frame[300:, :] = [35, 55, 35]
        
        # Add terrain texture
        for _ in range(100):
            x = np.random.randint(0, self.width)
            y = np.random.randint(300, self.height)
            cv2.circle(frame, (x, y), np.random.randint(2, 8), (40, 65, 40), -1)
        
        # Draw border fence (background)
        cv2.line(frame, (50, 300), (self.width - 50, 300), (100, 100, 100), 3)
        for x in range(50, self.width, 100):
            cv2.line(frame, (x, 280), (x, 320), (100, 100, 100), 2)
        
        # Draw virtual fence
        cv2.polylines(frame, [self.fence_points], True, (0, 255, 0), 3)
        
        # Fence label with background
        label = "VIRTUAL FENCE - ZONE-1"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (350 - 5, 235), (350 + tw + 5, 235 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, label, (350, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update and draw objects
        self._update_objects()
        self._draw_objects(frame)
        
        # Draw HUD
        self._draw_hud(frame)
        
        # Simulate signal loss event at 30 seconds
        if 28 < self.time < 35:
            self.cameras_online["CAM-03"] = False
        else:
            self.cameras_online["CAM-03"] = True
        
        return True, frame
    
    def _update_objects(self):
        """Update object positions"""
        for obj in self.objects:
            if not obj["active"]:
                continue
            
            # Update position
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            
            # Boundary checks
            margin = 80
            if obj["x"] < margin or obj["x"] > self.width - margin:
                obj["vx"] = -obj["vx"]
            if obj["y"] < 150 or obj["y"] > self.height - margin:
                obj["vy"] = -obj["vy"]
            
            # Check if inside virtual fence
            obj["in_fence"] = self._point_in_fence(obj["x"], obj["y"])
    
    def _point_in_fence(self, x: float, y: float) -> bool:
        """Check if point is inside virtual fence"""
        return cv2.pointPolygonTest(self.fence_points, (float(x), float(y)), False) >= 0
    
    def _draw_objects(self, frame: np.ndarray):
        """Draw all objects on frame"""
        for obj in self.objects:
            if not obj["active"]:
                continue
            
            x, y = int(obj["x"]), int(obj["y"])
            
            if obj["class"] == "person":
                # Person color based on fence status
                color = (0, 0, 255) if obj["in_fence"] else (0, 200, 255)
                
                # Draw person
                cv2.circle(frame, (x, y - 45), 12, color, -1)  # Head
                cv2.line(frame, (x, y - 33), (x, y + 15), color, 3)  # Body
                cv2.line(frame, (x, y - 10), (x - 15, y + 5), color, 3)  # Left arm
                cv2.line(frame, (x, y - 10), (x + 15, y + 5), color, 3)  # Right arm
                cv2.line(frame, (x, y + 15), (x - 12, y + 40), color, 3)  # Left leg
                cv2.line(frame, (x, y + 15), (x + 12, y + 40), color, 3)  # Right leg
                
                # Bounding box
                box_color = (0, 0, 255) if obj["in_fence"] else (0, 255, 255)
                cv2.rectangle(frame, (x - 30, y - 60), (x + 30, y + 50), box_color, 2)
                
                # Label
                label = f"Person {'WARNING' if obj['in_fence'] else 'OK'}"
                conf = 85 + int(self.time * 2) % 15
                cv2.putText(frame, f"{label} {conf}%", (x - 30, y - 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
            
            elif obj["class"] == "vehicle":
                # Vehicle
                color = (120, 120, 160)
                cv2.rectangle(frame, (x - 65, y - 25), (x + 65, y + 25), color, -1)
                cv2.rectangle(frame, (x - 45, y - 40), (x + 45, y - 25), (90, 90, 130), -1)
                
                # Wheels
                cv2.circle(frame, (x - 45, y + 25), 10, (30, 30, 30), -1)
                cv2.circle(frame, (x + 45, y + 25), 10, (30, 30, 30), -1)
                
                # License plate
                cv2.rectangle(frame, (x - 30, y + 5), (x + 30, y + 22), (255, 255, 255), -1)
                cv2.putText(frame, obj["plate"], (x - 27, y + 19),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Bounding box
                cv2.rectangle(frame, (x - 75, y - 50), (x + 75, y + 35), (255, 165, 0), 2)
                conf = 88 + int(self.time) % 12
                cv2.putText(frame, f"Vehicle {conf}%", (x - 75, y - 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
    
    def _draw_hud(self, frame: np.ndarray):
        """Draw heads-up display"""
        # Header
        cv2.rectangle(frame, (0, 0), (self.width, 55), (15, 15, 25), -1)
        cv2.line(frame, (0, 55), (self.width, 55), (88, 166, 255), 2)
        
        # Logo
        cv2.putText(frame, "IBVAP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (88, 166, 255), 3)
        cv2.putText(frame, "Intelligent Border Video Analytics", (160, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 130, 140), 1)
        
        # Recording indicator
        rec_color = (0, 0, 255) if int(self.time * 2) % 2 == 0 else (0, 0, 100)
        cv2.circle(frame, (self.width - 100, 30), 8, rec_color, -1)
        cv2.putText(frame, "REC", (self.width - 85, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (self.width - 280, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Camera status panel
        panel_y = 70
        cv2.rectangle(frame, (10, panel_y), (200, panel_y + 90), (20, 20, 30), -1)
        cv2.putText(frame, "CAMERAS", (20, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 166, 255), 1)
        
        for i, (cam, online) in enumerate(self.cameras_online.items()):
            y = panel_y + 40 + i * 20
            status_color = (0, 200, 0) if online else (0, 0, 255)
            cv2.circle(frame, (25, y), 5, status_color, -1)
            cv2.putText(frame, cam, (35, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Stats panel
        stats_y = 70
        cv2.rectangle(frame, (self.width - 200, stats_y), (self.width - 10, stats_y + 100), (20, 20, 30), -1)
        
        stats = [
            f"FPS: {self.fps}",
            f"Objects: {len([o for o in self.objects if o['active']])}",
            f"Frame: {self.frame_count}",
            f"Time: {self.time:.1f}s"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (self.width - 190, stats_y + 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    def get_events(self) -> List[Dict]:
        """Get current events for alert generation"""
        events = []
        
        # Check fence intrusions
        for obj in self.objects:
            if obj["class"] == "person" and obj["in_fence"]:
                events.append({
                    "type": "fence_intrusion",
                    "object": obj,
                    "severity": "high",
                    "explanation": f"Person T-{obj['id']:04d} crossed virtual fence Zone-1"
                })
        
        # Check signal loss
        if not self.cameras_online.get("CAM-03", True):
            events.append({
                "type": "signal_loss",
                "camera": "CAM-03",
                "severity": "critical",
                "explanation": "Camera CAM-03 signal lost for 3+ seconds"
            })
        
        return events


# ============================================================
# Alert System
# ============================================================

class AlertSystem:
    """Generate and manage alerts with hash chain"""
    
    def __init__(self):
        self.alerts = []
        self.prev_hash = "0" * 64
    
    def create_alert(self, event: Dict) -> Dict:
        """Create a new alert"""
        alert = {
            "event_id": f"e{len(self.alerts) + 1:04d}",
            "prev_hash": self.prev_hash,
            "timestamp": datetime.now().isoformat(),
            "site_id": "BOP-01",
            "camera_id": "CAM-01",
            "event_type": event["type"],
            "object_class": event.get("object", {}).get("class", "none"),
            "track_id": f"T-{event.get('object', {}).get('id', 0):04d}",
            "zone": "Zone-1",
            "confidence": 0.91,
            "severity": event["severity"],
            "explanation": event["explanation"]
        }
        
        # Update hash chain
        alert_copy = alert.copy()
        alert_copy.pop("prev_hash", None)
        self.prev_hash = hashlib.sha256(json.dumps(alert_copy, sort_keys=True).encode()).hexdigest()
        
        self.alerts.append(alert)
        return alert
    
    def verify_chain(self) -> bool:
        """Verify hash chain integrity"""
        for i in range(1, len(self.alerts)):
            prev_event = self.alerts[i - 1].copy()
            prev_event.pop("prev_hash", None)
            calculated_hash = hashlib.sha256(json.dumps(prev_event, sort_keys=True).encode()).hexdigest()
            if calculated_hash != self.alerts[i]["prev_hash"]:
                return False
        return True


# ============================================================
# Main Demo
# ============================================================

class IBVAPDemo:
    """Complete IBVAP demo for SIH presentation"""
    
    def __init__(self):
        self.config = DemoConfig()
        self.scene = BorderSceneGenerator(self.config.width, self.config.height)
        self.alerts = AlertSystem()
        self.frame_count = 0
        self.start_time = None
    
    def run(self):
        """Run the complete demo"""
        print("\n" + "=" * 70)
        print("  IBVAP - Intelligent Border Video Analytics Platform")
        print("  SIH 2026 Demo")
        print("=" * 70)
        print("\nControls:")
        print("  q - Quit demo")
        print("  f - Toggle fullscreen")
        print("  a - Show alert log")
        print("  s - Simulate signal loss")
        print("=" * 70 + "\n")
        
        self.start_time = time.time()
        
        # Create window
        cv2.namedWindow("IBVAP Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IBVAP Demo", self.config.width, self.config.height)
        
        while True:
            # Generate frame
            ret, frame = self.scene.generate_frame()
            if not ret:
                continue
            
            self.frame_count += 1
            
            # Generate alerts
            events = self.scene.get_events()
            for event in events:
                alert = self.alerts.create_alert(event)
            
            # Draw alert panel
            display = self._draw_alert_panel(frame)
            
            # Draw demo info
            display = self._draw_demo_info(display)
            
            # Show frame
            cv2.imshow("IBVAP Demo", display)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                cv2.setWindowProperty("IBVAP Demo", cv2.WND_PROP_FULLSCREEN,
                                     cv2.WINDOW_FULLSCREEN)
            elif key == ord('a'):
                self._show_alert_log()
            elif key == ord('s'):
                # Toggle signal loss
                self.scene.cameras_online["CAM-03"] = not self.scene.cameras_online["CAM-03"]
        
        self._cleanup()
    
    def _draw_alert_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw alert panel on right side"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Panel background
        panel_w = 350
        cv2.rectangle(display, (w - panel_w, 55), (w, h), (15, 15, 25), -1)
        cv2.line(display, (w - panel_w, 55), (w - panel_w, h), (88, 166, 255), 2)
        
        # Panel header
        cv2.putText(display, "ALERT LOG", (w - panel_w + 15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)
        
        # Draw recent alerts
        recent = self.alerts.alerts[-6:]  # Last 6 alerts
        y = 115
        
        for alert in reversed(recent):
            severity = alert["severity"]
            color = {
                "critical": (156, 39, 176),
                "high": (244, 67, 54),
                "medium": (255, 152, 0),
                "low": (76, 175, 80)
            }.get(severity, (100, 100, 100))
            
            # Alert box
            cv2.rectangle(display, (w - panel_w + 10, y), (w - 10, y + 55), color, 2)
            
            # Alert text
            cv2.putText(display, alert["event_type"].upper(),
                       (w - panel_w + 20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display, alert["explanation"][:40],
                       (w - panel_w + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            cv2.putText(display, alert["timestamp"][:19],
                       (w - panel_w + 20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
            
            y += 65
        
        # Hash chain status
        chain_valid = self.alerts.verify_chain()
        chain_color = (0, 200, 0) if chain_valid else (0, 0, 255)
        chain_text = "CHAIN: VALID" if chain_valid else "CHAIN: TAMPERED"
        
        cv2.rectangle(display, (w - panel_w + 10, h - 45), (w - 10, h - 10), (25, 25, 35), -1)
        cv2.putText(display, chain_text, (w - panel_w + 20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, chain_color, 1)
        
        return display
    
    def _draw_demo_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw demo information overlay"""
        display = frame.copy()
        
        # Demo timer
        elapsed = time.time() - self.start_time
        timer_text = f"Demo Time: {elapsed:.1f}s"
        cv2.putText(display, timer_text, (20, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def _show_alert_log(self):
        """Show full alert log in console"""
        print("\n" + "=" * 70)
        print("  IBVAP ALERT LOG")
        print("=" * 70)
        
        for i, alert in enumerate(self.alerts.alerts):
            print(f"\n[{i + 1}] {alert['event_type'].upper()}")
            print(f"    Time: {alert['timestamp']}")
            print(f"    Zone: {alert['zone']}")
            print(f"    Severity: {alert['severity'].upper()}")
            print(f"    Explanation: {alert['explanation']}")
            print(f"    Hash: {alert['prev_hash'][:16]}...")
        
        print("\n" + "=" * 70)
        chain_valid = self.alerts.verify_chain()
        print(f"Hash Chain Status: {'VALID ✓' if chain_valid else 'TAMPERED ✗'}")
        print(f"Total Alerts: {len(self.alerts.alerts)}")
        print("=" * 70 + "\n")
    
    def _cleanup(self):
        """Cleanup and print final stats"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("  IBVAP Demo Complete")
        print("=" * 70)
        print(f"\n  Duration: {elapsed:.1f} seconds")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Average FPS: {self.frame_count / elapsed:.1f}")
        print(f"  Total Alerts: {len(self.alerts.alerts)}")
        
        # Count by severity
        severity_count = {}
        for alert in self.alerts.alerts:
            sev = alert["severity"]
            severity_count[sev] = severity_count.get(sev, 0) + 1
        
        print("\n  Alerts by Severity:")
        for sev, count in severity_count.items():
            print(f"    {sev.upper()}: {count}")
        
        # Hash chain
        chain_valid = self.alerts.verify_chain()
        print(f"\n  Hash Chain: {'VALID ✓' if chain_valid else 'TAMPERED ✗'}")
        print("=" * 70 + "\n")
        
        cv2.destroyAllWindows()


# ============================================================
# Entry Point
# ============================================================

def main():
    demo = IBVAPDemo()
    demo.run()


if __name__ == "__main__":
    main()
