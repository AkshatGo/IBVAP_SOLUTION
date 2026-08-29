"""
IBVAP Demo Recording Script
Automated demo video generation for SIH 2026 presentation
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class DemoRecorder:
    """
    Automated demo video recorder for IBVAP
    Creates professional demo videos for hackathon presentation
    """
    
    def __init__(self, output_dir: str = "demo_videos", fps: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fps = fps
        self.current_scene = None
        
        # Video writers
        self.writers = {}
        
        # Demo configuration
        self.config = {
            "title_duration": 3,  # seconds
            "scene_duration": 15,  # seconds per scene
            "transition_duration": 1,  # seconds
            "resolution": (1920, 1080),
        }
        
        print(f"[DemoRecorder] Initialized")
        print(f"[DemoRecorder] Output directory: {self.output_dir}")
        print(f"[DemoRecorder] Resolution: {self.config['resolution']}")
    
    def create_title_frame(self, text: str, subtitle: str = "") -> np.ndarray:
        """Create a professional title frame"""
        width, height = self.config["resolution"]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dark gradient background
        for y in range(height):
            gradient = int(20 + (y / height) * 30)
            frame[y, :] = [gradient, gradient, gradient]
        
        # Title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        
        # Calculate text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw title
        x = (width - text_width) // 2
        y = (height - text_height) // 2 - 30
        cv2.putText(frame, text, (x, y), font, font_scale, (88, 166, 255), thickness)
        
        # Draw subtitle
        if subtitle:
            sub_scale = 1.0
            sub_thickness = 2
            (sub_width, _), _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)
            sub_x = (width - sub_width) // 2
            sub_y = y + text_height + 40
            cv2.putText(frame, subtitle, (sub_x, sub_y), font, sub_scale, (139, 148, 158), sub_thickness)
        
        # Draw accent line
        line_y = y + text_height + 60
        cv2.line(frame, (width // 2 - 100, line_y), (width // 2 + 100, line_y), (88, 166, 255), 2)
        
        return frame
    
    def create_scene_frame(
        self,
        scene_name: str,
        content: np.ndarray,
        alert_text: str = "",
        stats: dict = None
    ) -> np.ndarray:
        """Create a scene frame with dashboard overlay"""
        width, height = self.config["resolution"]
        
        # Resize content to fit
        content_height = height - 150  # Leave space for header
        content_width = int(content.shape[1] * (content_height / content.shape[0]))
        content = cv2.resize(content, (min(content_width, width), content_height))
        
        # Create frame with dark background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [13, 17, 23]  # Dark background
        
        # Draw header bar
        cv2.rectangle(frame, (0, 0), (width, 60), (22, 27, 34), -1)
        
        # Draw logo text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "IBVAP", (20, 40), font, 1.2, (88, 166, 255), 2)
        cv2.putText(frame, "Intelligent Border Video Analytics", (150, 40), font, 0.7, (139, 148, 158), 1)
        
        # Draw scene name
        cv2.putText(frame, scene_name, (width - 300, 40), font, 0.8, (255, 255, 255), 1)
        
        # Center content
        x_offset = (width - content.shape[1]) // 2
        y_offset = 80
        
        # Ensure content fits within frame bounds
        if x_offset + content.shape[1] > width:
            content = content[:, :width - x_offset]
        if y_offset + content.shape[0] > height:
            content = content[:height - y_offset, :]
        
        frame[y_offset:y_offset + content.shape[0], x_offset:x_offset + content.shape[1]] = content
        
        # Draw alert banner if provided
        if alert_text:
            banner_y = height - 120
            cv2.rectangle(frame, (0, banner_y), (width, banner_y + 50), (244, 67, 54), -1)
            cv2.putText(frame, alert_text, (20, banner_y + 35), font, 0.8, (255, 255, 255), 2)
        
        # Draw stats panel
        if stats:
            panel_x = width - 350
            panel_y = 80
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 330, panel_y + 200), (30, 30, 46), -1)
            
            for i, (key, value) in enumerate(stats.items()):
                y = panel_y + 30 + i * 35
                cv2.putText(frame, f"{key}:", (panel_x + 10, y), font, 0.5, (139, 148, 158), 1)
                cv2.putText(frame, str(value), (panel_x + 150, y), font, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def create_virtual_fence_demo(self) -> list:
        """Create virtual fence intrusion demo frames"""
        frames = []
        width, height = self.config["resolution"]
        
        # Simulate a border scene with fence
        for i in range(self.fps * 15):  # 15 seconds
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw grid pattern (simulating camera view)
            for x in range(0, width, 50):
                cv2.line(frame, (x, 0), (x, height), (40, 40, 40), 1)
            for y in range(0, height, 50):
                cv2.line(frame, (0, y), (width, y), (40, 40, 40), 1)
            
            # Draw virtual fence polygon
            fence_points = np.array([
                [300, 200], [900, 200], [900, 700], [300, 700]
            ], np.int32)
            cv2.polylines(frame, [fence_points], True, (0, 255, 0), 3)
            
            # Draw fence label
            cv2.putText(frame, "VIRTUAL FENCE", (350, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Animate person entering
            progress = i / (self.fps * 15)
            person_x = int(100 + progress * 600)
            person_y = int(450 + 50 * np.sin(progress * 10))
            
            # Draw person (simplified)
            cv2.circle(frame, (person_x, person_y - 50), 20, (0, 200, 255), -1)  # Head
            cv2.line(frame, (person_x, person_y - 30), (person_x, person_y + 30), (0, 200, 255), 3)  # Body
            cv2.line(frame, (person_x, person_y), (person_x - 20, person_y + 40), (0, 200, 255), 3)  # Left leg
            cv2.line(frame, (person_x, person_y), (person_x + 20, person_y + 40), (0, 200, 255), 3)  # Right leg
            
            # Detection box
            cv2.rectangle(frame, (person_x - 40, person_y - 70), (person_x + 40, person_y + 50), (0, 255, 255), 2)
            cv2.putText(frame, f"Person {91}%", (person_x - 40, person_y - 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Check if person crossed fence
            if person_x > 300:
                # Alert banner
                cv2.rectangle(frame, (0, height - 100), (width, height), (244, 67, 54), -1)
                cv2.putText(frame, "FENCE INTRUSION DETECTED", (20, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "Track T-0042 crossed Zone-3 at 1.4 m/s, bearing NE", (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
            
            frames.append(frame)
        
        return frames
    
    def create_anpr_demo(self) -> list:
        """Create ANPR multi-frame consensus demo frames"""
        frames = []
        width, height = self.config["resolution"]
        
        plates = ["BR12AB3456", "BR12AB3458", "BR12AB3456", "BR12AB3456", "BR12AB3457"]
        
        for i in range(self.fps * 15):  # 15 seconds
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw camera view background
            frame[:] = [30, 30, 40]
            
            # Draw vehicle
            vehicle_x = int(200 + (i / (self.fps * 15)) * 800)
            vehicle_y = 400
            
            # Vehicle body
            cv2.rectangle(frame, (vehicle_x - 80, vehicle_y - 30), (vehicle_x + 80, vehicle_y + 30), (100, 100, 120), -1)
            cv2.rectangle(frame, (vehicle_x - 60, vehicle_y - 50), (vehicle_x + 60, vehicle_y - 30), (80, 80, 100), -1)
            
            # License plate
            plate_x = vehicle_x - 40
            plate_y = vehicle_y + 10
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 80, plate_y + 25), (255, 255, 255), -1)
            
            # Show OCR results
            frame_idx = min(i // (self.fps * 3), len(plates) - 1)
            current_plate = plates[frame_idx]
            
            cv2.putText(frame, current_plate, (plate_x + 5, plate_y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw OCR panel on right side
            panel_x = width - 400
            panel_y = 100
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 380, panel_y + 350), (20, 20, 30), -1)
            cv2.putText(frame, "MULTI-FRAME OCR", (panel_x + 20, panel_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2)
            
            # Show frame results
            for j in range(min(frame_idx + 1, 5)):
                y = panel_y + 70 + j * 40
                plate_text = plates[j] if j < len(plates) else "---"
                cv2.putText(frame, f"Frame {j+1}:", (panel_x + 20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)
                cv2.putText(frame, plate_text, (panel_x + 120, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show consensus
            if frame_idx >= 2:
                cv2.putText(frame, "CONSENSUS:", (panel_x + 20, panel_y + 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "BR12AB3456", (panel_x + 20, panel_y + 320),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def create_signal_loss_demo(self) -> list:
        """Create signal loss detection demo frames"""
        frames = []
        width, height = self.config["resolution"]
        
        for i in range(self.fps * 10):  # 10 seconds
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if i < self.fps * 7:  # First 7 seconds - normal operation
                # Normal camera view
                frame[:] = [40, 50, 60]
                
                # Draw camera grid
                for x in range(0, width, 100):
                    cv2.line(frame, (x, 0), (x, height), (60, 70, 80), 1)
                for y in range(0, height, 100):
                    cv2.line(frame, (0, y), (width, y), (60, 70, 80), 1)
                
                # Camera status
                cv2.circle(frame, (50, 50), 15, (0, 255, 0), -1)
                cv2.putText(frame, "CAM-01 ONLINE", (75, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (width - 250, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
            else:  # Signal loss
                # Static/noise effect
                noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
                frame = noise
                
                # Camera status - OFFLINE
                cv2.circle(frame, (50, 50), 15, (0, 0, 255), -1)
                cv2.putText(frame, "CAM-01 OFFLINE", (75, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Alert banner
                cv2.rectangle(frame, (0, height - 120), (width, height), (156, 39, 176), -1)
                cv2.putText(frame, "SIGNAL LOSS DETECTED", (20, height - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "Camera CAM-01 signal lost for 3 seconds - CRITICAL", (20, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 1)
            
            frames.append(frame)
        
        return frames
    
    def create_hash_chain_demo(self) -> list:
        """Create tamper-evident hash chain demo frames"""
        frames = []
        width, height = self.config["resolution"]
        
        # Create events
        events = []
        prev_hash = "0" * 16
        for i in range(5):
            event = {
                "id": f"e{i:04d}",
                "prev_hash": prev_hash,
                "type": "fence_intrusion",
                "data": f"Event {i}"
            }
            import hashlib
            prev_hash = hashlib.sha256(json.dumps(event).encode()).hexdigest()[:16]
            events.append(event)
        
        for i in range(self.fps * 10):  # 10 seconds
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [20, 20, 30]
            
            # Title
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "TAMPER-EVIDENT HASH CHAIN", (width // 2 - 300, 50),
                       font, 1.0, (88, 166, 255), 2)
            
            # Draw events
            event_y = 120
            for j, event in enumerate(events[:min(i // (self.fps * 2) + 1, 5)]):
                # Event box
                box_x = 100
                box_y = event_y + j * 120
                
                # Highlight current event
                if j == min(i // (self.fps * 2), 4):
                    cv2.rectangle(frame, (box_x, box_y), (box_x + 800, box_y + 100), (88, 166, 255), 2)
                else:
                    cv2.rectangle(frame, (box_x, box_y), (box_x + 800, box_y + 100), (60, 60, 80), 1)
                
                # Event info
                cv2.putText(frame, f"Event {event['id']}", (box_x + 20, box_y + 30),
                           font, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Prev Hash: {event['prev_hash']}", (box_x + 20, box_y + 60),
                           font, 0.4, (139, 148, 158), 1)
                cv2.putText(frame, f"Type: {event['type']}", (box_x + 20, box_y + 85),
                           font, 0.4, (0, 255, 0), 1)
                
                # Arrow to next event
                if j < 4 and j < len(events) - 1:
                    cv2.arrowedLine(frame, (box_x + 400, box_y + 100), 
                                   (box_x + 400, box_y + 120), (88, 166, 255), 2)
            
            # Verification status
            if i > self.fps * 5:
                status_y = height - 100
                cv2.rectangle(frame, (100, status_y), (width - 100, status_y + 60), (0, 100, 0), -1)
                cv2.putText(frame, "CHAIN VERIFIED: ALL EVENTS VALID", (150, status_y + 40),
                           font, 0.8, (0, 255, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def save_video(self, frames: list, filename: str):
        """Save frames as video"""
        if not frames:
            print("[DemoRecorder] No frames to save")
            return
        
        output_path = self.output_dir / filename
        width, height = self.config["resolution"]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        for frame in frames:
            # Ensure frame is correct size
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        
        writer.release()
        print(f"[DemoRecorder] Saved: {output_path}")
    
    def generate_all_demos(self):
        """Generate all demo videos"""
        print("[DemoRecorder] Generating demo videos...")
        
        # 1. Title screen
        print("  Generating title screen...")
        title_frame = self.create_title_frame("IBVAP", "Intelligent Border Video Analytics Platform")
        title_frames = [title_frame] * (self.fps * self.config["title_duration"])
        self.save_video(title_frames, "01_title.mp4")
        
        # 2. Virtual fence demo
        print("  Generating virtual fence demo...")
        fence_frames = self.create_virtual_fence_demo()
        self.save_video(fence_frames, "02_virtual_fence.mp4")
        
        # 3. ANPR demo
        print("  Generating ANPR demo...")
        anpr_frames = self.create_anpr_demo()
        self.save_video(anpr_frames, "03_anpr_consensus.mp4")
        
        # 4. Signal loss demo
        print("  Generating signal loss demo...")
        signal_frames = self.create_signal_loss_demo()
        self.save_video(signal_frames, "04_signal_loss.mp4")
        
        # 5. Hash chain demo
        print("  Generating hash chain demo...")
        hash_frames = self.create_hash_chain_demo()
        self.save_video(hash_frames, "05_hash_chain.mp4")
        
        print("[DemoRecorder] All demos generated!")
        print(f"[DemoRecorder] Videos saved to: {self.output_dir}")


def main():
    """Main function to run demo recording"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IBVAP Demo Video Generator")
    parser.add_argument("--output", default="demo_videos", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--scene", choices=["all", "fence", "anpr", "signal", "hash"], 
                       default="all", help="Scene to generate")
    
    args = parser.parse_args()
    
    recorder = DemoRecorder(output_dir=args.output, fps=args.fps)
    
    if args.scene == "all":
        recorder.generate_all_demos()
    elif args.scene == "fence":
        frames = recorder.create_virtual_fence_demo()
        recorder.save_video(frames, "virtual_fence.mp4")
    elif args.scene == "anpr":
        frames = recorder.create_anpr_demo()
        recorder.save_video(frames, "anpr_consensus.mp4")
    elif args.scene == "signal":
        frames = recorder.create_signal_loss_demo()
        recorder.save_video(frames, "signal_loss.mp4")
    elif args.scene == "hash":
        frames = recorder.create_hash_chain_demo()
        recorder.save_video(frames, "hash_chain.mp4")


if __name__ == "__main__":
    main()
