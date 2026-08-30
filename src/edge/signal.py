"""
Signal Loss Detector — Alerts when camera feeds are lost or tampered.
Critical for border surveillance: a blinded camera is itself an alert.

Thresholds are read from src.config so they can be tuned live during
a demo without redeploying (e.g. via a config file or environment
variables that override the dataclass defaults).
"""
import os
import json
import time
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


# Optional: load overrides from config/signal_thresholds.json
_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "signal_thresholds.json"
)


def _load_overrides() -> dict:
    """Load threshold overrides from JSON file if it exists."""
    if os.path.exists(_THRESHOLDS_PATH):
        try:
            with open(_THRESHOLDS_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


@dataclass
class CameraStatus:
    """Status of a single camera."""
    camera_id: str
    is_online: bool = True
    last_frame_time: float = 0.0
    signal_loss_time: Optional[float] = None
    frame_count: int = 0
    avg_brightness: float = 128.0
    consecutive_black: int = 0

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "is_online": self.is_online,
            "last_frame_time": self.last_frame_time,
            "signal_loss_time": self.signal_loss_time,
            "frame_count": self.frame_count,
            "avg_brightness": round(self.avg_brightness, 1),
            "consecutive_black": self.consecutive_black,
        }


class SignalLossDetector:
    """
    Detects camera signal loss via multiple heuristics:
    1. No frame received for N seconds
    2. Frame is all-black (possible tampering)
    3. Frame is all-white (possible sensor failure)
    4. Sudden brightness drop (possible jamming)

    Thresholds can be overridden at runtime by writing to
    config/signal_thresholds.json:
    {
      "timeout_seconds": 3.0,
      "black_threshold": 5.0,
      "white_threshold": 250.0,
      "brightness_drop_threshold": 0.7,
      "consecutive_black_needed": 3
    }
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        black_threshold: float = 5.0,
        white_threshold: float = 250.0,
        brightness_drop_threshold: float = 0.7,
        consecutive_black_needed: int = 3,
    ):
        overrides = _load_overrides()
        self.timeout_seconds = overrides.get("timeout_seconds", timeout_seconds)
        self.black_threshold = overrides.get("black_threshold", black_threshold)
        self.white_threshold = overrides.get("white_threshold", white_threshold)
        self.brightness_drop_threshold = overrides.get(
            "brightness_drop_threshold", brightness_drop_threshold
        )
        self.consecutive_black_needed = overrides.get(
            "consecutive_black_needed", consecutive_black_needed
        )
        self.cameras: Dict[str, CameraStatus] = {}

    def reload_thresholds(self):
        """Hot-reload thresholds from config file without restarting."""
        overrides = _load_overrides()
        if "timeout_seconds" in overrides:
            self.timeout_seconds = overrides["timeout_seconds"]
        if "black_threshold" in overrides:
            self.black_threshold = overrides["black_threshold"]
        if "white_threshold" in overrides:
            self.white_threshold = overrides["white_threshold"]
        if "brightness_drop_threshold" in overrides:
            self.brightness_drop_threshold = overrides["brightness_drop_threshold"]
        if "consecutive_black_needed" in overrides:
            self.consecutive_black_needed = overrides["consecutive_black_needed"]

    def register_camera(self, camera_id: str):
        """Register a camera for monitoring."""
        self.cameras[camera_id] = CameraStatus(
            camera_id=camera_id,
            last_frame_time=time.time(),
        )

    def update(self, camera_id: str, frame: np.ndarray) -> Optional[str]:
        """
        Update camera status with new frame.
        Returns alert message if signal loss detected, None otherwise.
        """
        now = time.time()

        if camera_id not in self.cameras:
            self.register_camera(camera_id)

        cam = self.cameras[camera_id]
        cam.frame_count += 1
        cam.last_frame_time = now

        # Check frame quality
        if frame is None or frame.size == 0:
            return self._trigger_loss(camera_id, cam, "empty_frame",
                                       "Camera returned empty frame")

        gray = frame if len(frame.shape) == 2 else np.mean(frame, axis=2)
        avg_brightness = float(np.mean(gray))
        # Capture the previous reading *before* overwriting it — the
        # brightness-drop check below compares the two, and comparing a
        # value against itself can never trip.
        prev_brightness = cam.avg_brightness
        cam.avg_brightness = avg_brightness

        # Check for all-black frame (possible jamming/cover)
        if avg_brightness < self.black_threshold:
            cam.consecutive_black += 1
            if cam.consecutive_black >= self.consecutive_black_needed:
                return self._trigger_loss(camera_id, cam, "black_frame",
                                           f"Camera showing black frames "
                                           f"(brightness={avg_brightness:.1f}). "
                                           f"Possible tampering or jamming.")
        else:
            cam.consecutive_black = 0

        # Check for all-white frame (sensor failure)
        if avg_brightness > self.white_threshold:
            return self._trigger_loss(camera_id, cam, "white_frame",
                                       f"Camera showing white frames "
                                       f"(brightness={avg_brightness:.1f}). "
                                       f"Possible sensor failure.")

        # Check for sudden brightness drop (possible gradual jamming)
        if cam.frame_count > 10:
            if prev_brightness > 0 and avg_brightness / prev_brightness < self.brightness_drop_threshold:
                return self._trigger_loss(camera_id, cam, "brightness_drop",
                                           f"Sudden brightness drop: "
                                           f"{prev_brightness:.0f} -> {avg_brightness:.0f}. "
                                           f"Possible interference.")

        # If camera was offline, restore it
        if not cam.is_online:
            cam.is_online = True
            cam.signal_loss_time = None
            return f"Camera {camera_id} signal restored."

        return None

    def check_timeout(self, camera_id: str) -> Optional[str]:
        """Check if camera has timed out (no frames received)."""
        if camera_id not in self.cameras:
            return None

        cam = self.cameras[camera_id]
        elapsed = time.time() - cam.last_frame_time

        if cam.is_online and elapsed > self.timeout_seconds:
            return self._trigger_loss(camera_id, cam, "timeout",
                                       f"No frame received for {elapsed:.1f}s. "
                                       f"Possible signal loss or camera failure.")

        return None

    def check_all_timeouts(self) -> list:
        """Check all cameras for timeout."""
        alerts = []
        for cam_id in self.cameras:
            alert = self.check_timeout(cam_id)
            if alert:
                alerts.append((cam_id, alert))
        return alerts

    def _trigger_loss(self, camera_id: str, cam: CameraStatus,
                      loss_type: str, message: str) -> str:
        """Internal: trigger a signal loss event."""
        if cam.is_online:
            cam.is_online = False
            cam.signal_loss_time = time.time()
        return f"[{loss_type.upper()}] {message}"

    def get_status(self) -> Dict[str, dict]:
        """Get status of all cameras."""
        return {cid: cam.to_dict() for cid, cam in self.cameras.items()}

    def get_online_count(self) -> int:
        """Count of online cameras."""
        return sum(1 for cam in self.cameras.values() if cam.is_online)

    def get_offline_count(self) -> int:
        """Count of offline cameras."""
        return sum(1 for cam in self.cameras.values() if not cam.is_online)

    def get_thresholds(self) -> dict:
        """Return current threshold values (for API display)."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "black_threshold": self.black_threshold,
            "white_threshold": self.white_threshold,
            "brightness_drop_threshold": self.brightness_drop_threshold,
            "consecutive_black_needed": self.consecutive_black_needed,
        }
