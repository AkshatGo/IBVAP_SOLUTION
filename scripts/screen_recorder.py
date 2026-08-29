"""
IBVAP Screen Recorder
Capture screen recordings for live demo presentation
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Try to import screen capture libraries
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from mss import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class ScreenRecorder:
    """
    Screen recording utility for capturing live demos
    Supports multiple backends: pyautogui, mss, or fallback to test pattern
    """
    
    def __init__(
        self,
        output_dir: str = "recordings",
        fps: int = 30,
        region: tuple = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fps = fps
        self.region = region  # (x, y, width, height) or None for full screen
        self.is_recording = False
        self.writer = None
        
        # Determine capture method
        if PYAUTOGUI_AVAILABLE:
            self.capture_method = "pyautogui"
        elif MSS_AVAILABLE:
            self.capture_method = "mss"
        else:
            self.capture_method = "test"
            print("[ScreenRecorder] Warning: No screen capture library available")
            print("[ScreenRecorder] Install pyautogui or mss for screen recording")
            print("[ScreenRecorder] Using test pattern mode")
        
        print(f"[ScreenRecorder] Initialized")
        print(f"[ScreenRecorder] Capture method: {self.capture_method}")
        print(f"[ScreenRecorder] Output directory: {self.output_dir}")
    
    def get_screen_size(self) -> tuple:
        """Get screen dimensions"""
        if self.capture_method == "pyautogui":
            return pyautogui.size()
        elif self.capture_method == "mss":
            with mss() as sct:
                monitor = sct.monitors[1]
                return (monitor['width'], monitor['height'])
        else:
            return (1920, 1080)
    
    def capture_screen(self) -> np.ndarray:
        """Capture current screen"""
        if self.capture_method == "pyautogui":
            if self.region:
                screenshot = pyautogui.screenshot(region=self.region)
            else:
                screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        elif self.capture_method == "mss":
            with mss() as sct:
                if self.region:
                    monitor = {
                        "top": self.region[1],
                        "left": self.region[0],
                        "width": self.region[2],
                        "height": self.region[3]
                    }
                else:
                    monitor = sct.monitors[1]
                
                screenshot = sct.grab(monitor)
                return cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
        
        else:
            # Test pattern mode
            return self._generate_test_pattern()
    
    def _generate_test_pattern(self) -> np.ndarray:
        """Generate test pattern for demo purposes"""
        width, height = 1920, 1080
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Grid pattern
        for x in range(0, width, 50):
            cv2.line(frame, (x, 0), (x, height), (40, 40, 40), 1)
        for y in range(0, height, 50):
            cv2.line(frame, (0, y), (width, y), (40, 40, 40), 1)
        
        # Title
        cv2.putText(frame, "IBVAP DEMO MODE", (width // 2 - 200, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (88, 166, 255), 3)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width // 2 - 150, height // 2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 148, 158), 2)
        
        return frame
    
    def start_recording(self, filename: str = None):
        """Start screen recording"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
        
        output_path = self.output_dir / filename
        width, height = self.get_screen_size()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        self.is_recording = True
        print(f"[ScreenRecorder] Recording started: {output_path}")
        print(f"[ScreenRecorder] Resolution: {width}x{height}")
        print(f"[ScreenRecorder] Press Ctrl+C to stop recording")
        
        return output_path
    
    def record_frame(self):
        """Capture and write a single frame"""
        if not self.is_recording or self.writer is None:
            return False
        
        frame = self.capture_screen()
        self.writer.write(frame)
        return True
    
    def stop_recording(self):
        """Stop screen recording"""
        self.is_recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        print("[ScreenRecorder] Recording stopped")
    
    def record_for_duration(self, duration: float, filename: str = None):
        """Record screen for a specific duration"""
        output_path = self.start_recording(filename)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.record_frame()
                frame_count += 1
                
                # Maintain FPS
                elapsed = time.time() - start_time
                expected_frames = elapsed * self.fps
                if frame_count > expected_frames:
                    time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[ScreenRecorder] Recording interrupted")
        finally:
            self.stop_recording()
        
        print(f"[ScreenRecorder] Recorded {frame_count} frames")
        return output_path
    
    def record_interactive(self, filename: str = None):
        """Record until user presses Ctrl+C"""
        output_path = self.start_recording(filename)
        
        frame_count = 0
        
        try:
            while self.is_recording:
                self.record_frame()
                frame_count += 1
                time.sleep(1 / self.fps)
                
        except KeyboardInterrupt:
            print("\n[ScreenRecorder] Stopping recording...")
        finally:
            self.stop_recording()
        
        print(f"[ScreenRecorder] Total frames recorded: {frame_count}")
        return output_path


class DemoPresenter:
    """
    Interactive demo presenter with timed scenes
    """
    
    def __init__(self, recorder: ScreenRecorder):
        self.recorder = recorder
        self.scenes = []
        self.current_scene = 0
    
    def add_scene(self, name: str, duration: float, description: str = ""):
        """Add a scene to the presentation"""
        self.scenes.append({
            "name": name,
            "duration": duration,
            "description": description
        })
    
    def run_presentation(self, filename: str = "demo_presentation.mp4"):
        """Run through all scenes and record"""
        if not self.scenes:
            print("[DemoPresenter] No scenes defined")
            return
        
        output_path = self.recorder.start_recording(filename)
        
        total_duration = sum(scene["duration"] for scene in self.scenes)
        print(f"[DemoPresenter] Starting presentation ({total_duration:.1f} seconds)")
        
        try:
            for i, scene in enumerate(self.scenes):
                print(f"\n[DemoPresenter] Scene {i + 1}/{len(self.scenes)}: {scene['name']}")
                print(f"[DemoPresenter] Duration: {scene['duration']} seconds")
                if scene['description']:
                    print(f"[DemoPresenter] {scene['description']}")
                
                # Record scene
                start_time = time.time()
                while time.time() - start_time < scene["duration"]:
                    self.recorder.record_frame()
                    time.sleep(1 / self.recorder.fps)
                
                print(f"[DemoPresenter] Scene {i + 1} complete")
        
        except KeyboardInterrupt:
            print("\n[DemoPresenter] Presentation interrupted")
        finally:
            self.recorder.stop_recording()
        
        print(f"\n[DemoPresenter] Presentation saved: {output_path}")
        return output_path


def record_ibvap_demo():
    """Record a complete IBVAP demo"""
    print("=" * 60)
    print("IBVAP Demo Recording")
    print("=" * 60)
    
    recorder = ScreenRecorder(output_dir="demo_recordings")
    presenter = DemoPresenter(recorder)
    
    # Add scenes
    presenter.add_scene(
        "Title Screen",
        5.0,
        "Show IBVAP title and one-liner"
    )
    presenter.add_scene(
        "Architecture Overview",
        30.0,
        "Explain three-tier architecture"
    )
    presenter.add_scene(
        "Virtual Fence Demo",
        60.0,
        "Draw fence, show intrusion detection"
    )
    presenter.add_scene(
        "ANPR Demo",
        60.0,
        "Show multi-frame OCR consensus"
    )
    presenter.add_scene(
        "Signal Loss Demo",
        45.0,
        "Kill camera feed, show alert"
    )
    presenter.add_scene(
        "Hash Chain Demo",
        30.0,
        "Show tamper-evident log verification"
    )
    
    # Run presentation
    presenter.run_presentation("ibvap_full_demo.mp4")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IBVAP Screen Recorder")
    parser.add_argument("--mode", choices=["interactive", "timed", "demo"], 
                       default="interactive", help="Recording mode")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration for timed mode (seconds)")
    parser.add_argument("--output", default="recording.mp4",
                       help="Output filename")
    parser.add_argument("--fps", type=int, default=30,
                       help="Recording FPS")
    parser.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                       help="Screen region to capture")
    
    args = parser.parse_args()
    
    # Create recorder
    region = tuple(args.region) if args.region else None
    recorder = ScreenRecorder(fps=args.fps, region=region)
    
    if args.mode == "interactive":
        print("Starting interactive recording...")
        print("Press Ctrl+C to stop")
        recorder.record_interactive(args.output)
    
    elif args.mode == "timed":
        print(f"Recording for {args.duration} seconds...")
        recorder.record_for_duration(args.duration, args.output)
    
    elif args.mode == "demo":
        record_ibvap_demo()


if __name__ == "__main__":
    main()
