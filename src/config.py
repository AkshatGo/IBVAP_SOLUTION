"""
IBVAP Core Configuration
All system-wide settings in one place.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("ibvap.config")

# Repo-root-relative, not cwd-relative: the demo, the API server and the
# scripts/ tools are all launched from different working directories, and a
# bare Path("models/...") resolves against whichever one is current.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTION_WEIGHTS_DEFAULT = _REPO_ROOT / "models" / "weights" / "detection.pt"
PLATE_WEIGHTS_DEFAULT = _REPO_ROOT / "models" / "weights" / "plate.pt"


@lru_cache(maxsize=None)
def _resolve_model_path(env_var: str, fine_tuned: Path, stock_fallback, label: str):
    """Pick the weights: env override -> fine-tuned on disk -> stock fallback.

    The fallback is never silent. Before this, `main.py demo` and the API
    server defaulted to stock COCO while web_demo.py loaded the fine-tuned
    weights by hand, so the same repo demoed two different models and
    nothing on screen said which. Cached so the choice is logged once per
    process rather than once per config object.
    """
    override = os.environ.get(env_var)
    if override:
        log.info("%s: env override %s -> %s", label, env_var, override)
        return override
    if override == "":
        # Set-but-empty is deliberate, and distinct from unset: it forces the
        # stock/no-model baseline so the fine-tuned side can be A/B'd against
        # it without moving the weights out of the way.
        log.info("%s: %s set empty -> forcing baseline %r",
                 label, env_var, stock_fallback)
        return stock_fallback

    if fine_tuned.exists():
        log.info("%s: fine-tuned weights %s", label, fine_tuned)
        return str(fine_tuned)

    log.warning(
        "%s: fine-tuned weights missing at %s - falling back to %r. "
        "Expect degraded accuracy. Restore models/weights/ or set %s.",
        label, fine_tuned, stock_fallback, env_var,
    )
    return stock_fallback


def _resolve_detection_conf() -> float:
    """Confidence follows the weights, because the right cutoff differs.

    Swept with scripts/evaluate.py threshold, the IDD model's F1 peaks at
    0.25; 0.45 was picked against stock COCO and costs the fine-tuned model
    5.5 points of recall, which a border post pays for in missed intruders.
    Defaulting the model without defaulting the threshold would just trade
    one entry-point mismatch for another (web_demo.py:150 does the same).
    """
    override = os.environ.get("IBVAP_DETECTION_CONF")
    if override:
        return float(override)
    fine_tuned = _resolve_model_path(
        "IBVAP_DETECTION_MODEL", DETECTION_WEIGHTS_DEFAULT, "yolov8n.pt", "Detector"
    ) != "yolov8n.pt"
    return 0.25 if fine_tuned else 0.45


@dataclass
class DetectionConfig:
    """Object detection settings.

    `model_path` prefers the IDD fine-tuned weights in models/weights/ and
    falls back to stock YOLOv8-nano with a warning. Point
    IBVAP_DETECTION_MODEL at any checkpoint to override, which is what
    makes a live stock-vs-fine-tuned A/B possible (ROADMAP §5.2):
        IBVAP_DETECTION_MODEL=yolov8n.pt python main.py demo
    """
    model_path: str = field(default_factory=lambda: _resolve_model_path(
        "IBVAP_DETECTION_MODEL", DETECTION_WEIGHTS_DEFAULT, "yolov8n.pt", "Detector"
    ))
    # Follows the weights — 0.25 fine-tuned, 0.45 stock. See
    # _resolve_detection_conf(). Override with IBVAP_DETECTION_CONF.
    confidence_threshold: float = field(default_factory=_resolve_detection_conf)
    nms_threshold: float = 0.5
    input_size: Tuple[int, int] = (640, 640)
    # COCO's indices: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7.
    # These are correct for stock weights and MUST stay COCO's, because
    # EdgeDetector.load() (src/edge/detector.py) rebuilds the filter from the
    # model's own `names` for any non-80-class checkpoint — the fine-tuned
    # model's real order is bus=4, truck=5, autorickshaw=6, and it is adopted
    # at load time, not configured here. Hardcoding the 7-class order instead
    # would leave stock COCO reading index 4 as airplane and 6 as train.
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
    # Trained single-class plate localizer, preferred when present:
    # measured F1 0.915 against the contour localizer's 0.194 on the 91
    # held-out plates. Falls back to contours (no model, no GPU —
    # ROADMAP §3.3.4) with a warning. IBVAP_PLATE_MODEL overrides; set it
    # to an empty string to force the contour baseline for an A/B.
    plate_model_path: Optional[str] = field(default_factory=lambda: _resolve_model_path(
        "IBVAP_PLATE_MODEL", PLATE_WEIGHTS_DEFAULT, None, "Plate localizer"
    ))
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
