"""
Edge Detection Engine — YOLOv8 with class filtering.
Runs inference on each frame and returns structured detections.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class Detection:
    """Single object detection."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "center": self.center,
        }


class EdgeDetector:
    """
    YOLOv8-based object detector for edge deployment.
    Filters to target classes (person, vehicle types).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        target_classes: Optional[List[int]] = None,
        input_size: tuple = (640, 640),
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.target_classes = target_classes or [0, 1, 2, 3, 5, 7]
        self.input_size = input_size
        self.model = None
        self.class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck"
        }

    def load(self):
        """Load the YOLOv8 model."""
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        self.model = YOLO(self.model_path)
        return self

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        Returns list of Detection objects filtered to target classes.
        """
        if self.model is None:
            self.load()

        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=self.input_size[0],
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in self.target_classes:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                detections.append(Detection(
                    track_id=-1,  # assigned by tracker
                    class_id=cls,
                    class_name=self.class_names.get(cls, f"class_{cls}"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                ))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on multiple frames."""
        return [self.detect(f) for f in frames]

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        """Filter to only person detections."""
        return [d for d in detections if d.class_id == 0]

    def get_vehicles(self, detections: List[Detection]) -> List[Detection]:
        """Filter to vehicle detections."""
        return [d for d in detections if d.class_id in [1, 2, 3, 5, 7]]
