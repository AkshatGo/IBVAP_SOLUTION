"""
IBVAP Core Configuration
All system-wide settings in one place.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os
from pathlib import Path


@dataclass
class DetectionConfig:
    """Object detection settings.

    `model_path` defaults to stock YOLOv8-nano. Point IBVAP_DETECTION_MODEL
    at a fine-tuned checkpoint to swap it without a code change, which is
    what makes a live stock-vs-fine-tuned A/B possible (ROADMAP §5.2):
        IBVAP_DETECTION_MODEL=runs/detect/ibvap_detection/weights/best.pt
    """
    model_path: str = field(
        default_factory=lambda: os.environ.get("IBVAP_DETECTION_MODEL", "yolov8n.pt")
    )
    # 0.45 was chosen against stock COCO weights. Swept against the
    # IDD-fine-tuned model (scripts/evaluate.py threshold), F1 peaks at
    # 0.25 — 0.45 costs 5.5 points of recall for precision this deployment
    # does not need. A missed intruder is worse than a false alert at a
    # border, so the lower cutoff is the right default once fine-tuned
    # weights are in use. Override with IBVAP_DETECTION_CONF.
    confidence_threshold: float = field(
        default_factory=lambda: float(os.environ.get("IBVAP_DETECTION_CONF", "0.45"))
    )
    nms_threshold: float = 0.5
    input_size: Tuple[int, int] = (640, 640)
    # Classes we care about: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
    target_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    class_names: dict = field(default_factory=lambda: {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck"
    })


@dataclass
class TrackingConfig:
    """Object tracker settings (greedy IoU tracker — see src/edge/tracker.py)."""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_hits: int = 3
    frame_rate: int = 30


@dataclass
class ANPRConfig:
    """ANPR pipeline settings."""
    ocr_engine: str = "easyocr"  # or "paddleocr"
    languages: List[str] = field(default_factory=lambda: ["en"])
    consensus_frames: int = 5  # Number of frames for voting
    min_confidence: float = 0.6
    plate_min_area: int = 1000
    plate_max_area: int = 50000
    # Trained single-class plate localizer. None keeps the classical
    # contour localizer, which needs no model and no GPU (ROADMAP §3.3.4).
    plate_model_path: Optional[str] = field(
        default_factory=lambda: os.environ.get("IBVAP_PLATE_MODEL") or None
    )
    plate_model_confidence: float = 0.35
    indian_plate_pattern: str = r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}"


@dataclass
class FenceConfig:
    """Virtual fence settings."""
    # Default fence polygon (relative to 640x480 frame)
    default_fence: List[Tuple[int, int]] = field(default_factory=lambda: [
        (180, 120), (460, 120), (460, 380), (180, 380)
    ])
    alert_cooldown_seconds: float = 5.0
    zone_names: dict = field(default_factory=lambda: {
        "zone_1": "Primary Perimeter",
        "zone_2": "Restricted Area",
        "zone_3": "Critical Zone"
    })


@dataclass
class AlertConfig:
    """Alert and notification settings."""
    severity_levels: dict = field(default_factory=lambda: {
        "low": 1, "medium": 2, "high": 3, "critical": 4
    })
    hash_algorithm: str = "sha256"
    retention_days: int = 90
    max_alerts_memory: int = 10000
    mqtt_topic_prefix: str = "ibvap/alerts"


@dataclass
class CameraConfig:
    """Camera and video settings."""
    default_resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    signal_loss_timeout_seconds: float = 5.0
    reconnect_attempts: int = 3


@dataclass
class ServerConfig:
    """Backend server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    database_url: str = "sqlite:///ibvap.db"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class IBVAPConfig:
    """Master configuration."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    anpr: ANPRConfig = field(default_factory=ANPRConfig)
    fence: FenceConfig = field(default_factory=FenceConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def save(self, path: str = "config/ibvap_config.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "config/ibvap_config.json"):
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            cfg = cls()
            for key, val in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
            return cfg
        return cls()


# Global config singleton
CONFIG = IBVAPConfig()
