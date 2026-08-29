"""
ANPR Engine — Automatic Number Plate Recognition with multi-frame consensus voting.
Extracts plate regions, runs OCR, and uses majority voting across consecutive frames.
"""
import re
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class PlateResult:
    """ANPR result for a single plate reading."""
    plate_text: str
    confidence: float
    bbox: tuple
    frame_id: int

    def to_dict(self) -> dict:
        return {
            "plate_text": self.plate_text,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "frame_id": self.frame_id,
        }


@dataclass
class ConsensusResult:
    """Multi-frame consensus result."""
    plate_text: str
    confidence: float
    num_frames: int
    votes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plate_text": self.plate_text,
            "confidence": round(self.confidence, 3),
            "num_frames": self.num_frames,
            "votes": self.votes,
        }


class ANPREngine:
    """
    Indian plate ANPR with multi-frame OCR consensus voting.
    
    Pipeline:
    1. Plate detection via contour analysis + aspect ratio filtering
    2. Plate region cropping and preprocessing
    3. OCR (EasyOCR)
    4. Indian plate pattern validation
    5. Multi-frame majority voting
    """

    INDIAN_PATTERN = re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}")

    def __init__(
        self,
        consensus_frames: int = 5,
        min_confidence: float = 0.6,
        ocr_languages: List[str] = None,
    ):
        self.consensus_frames = consensus_frames
        self.min_confidence = min_confidence
        self.ocr_languages = ocr_languages or ["en"]
        self.reader = None
        self.readings: Dict[int, List[PlateResult]] = {}  # track_id -> readings
        self.consensus_results: Dict[int, ConsensusResult] = {}

    def load(self):
        """Load OCR engine."""
        try:
            import easyocr
            self.reader = easyocr.Reader(self.ocr_languages, gpu=False)
        except ImportError:
            print("[ANPR] easyocr not installed, OCR disabled")
            self.reader = None
        return self

    def detect_plate_region(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect candidate plate regions using contour analysis.
        Returns list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # quadrilateral
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                area = w * h
                # Indian plates are roughly 3:1 to 5:1 aspect ratio
                if 2.0 < aspect_ratio < 6.0 and 1000 < area < 50000:
                    plate_candidates.append((x, y, w, h))

        return plate_candidates

    def preprocess_plate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract and preprocess plate region for OCR."""
        x, y, w, h = bbox
        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        plate_img = frame[y1:y2, x1:x2]

        # Resize for better OCR
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(plate_img.shape) == 3:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Threshold
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return plate_img

    def read_plate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """Run OCR on a single plate region."""
        if self.reader is None:
            return None

        plate_img = self.preprocess_plate(frame, bbox)
        results = self.reader.readtext(plate_img)

        if not results:
            return None

        # Take the best result
        best_text = ""
        best_conf = 0.0
        for ( _, text, conf) in results:
            text = text.strip().upper()
            if conf > best_conf and len(text) >= 4:
                best_text = text
                best_conf = conf

        if best_conf < self.min_confidence or not best_text:
            return None

        # Validate Indian plate pattern
        cleaned = re.sub(r"[^A-Z0-9]", "", best_text)
        if not self.INDIAN_PATTERN.match(cleaned):
            # Try to fix common OCR errors
            cleaned = cleaned.replace("O", "0").replace("I", "1")
            cleaned = cleaned.replace("Z", "2").replace("S", "5")

        return PlateResult(
            plate_text=cleaned,
            confidence=best_conf,
            bbox=bbox,
            frame_id=0,
        )

    def add_reading(self, track_id: int, result: PlateResult):
        """Add a plate reading for a tracked vehicle."""
        if track_id not in self.readings:
            self.readings[track_id] = []
        self.readings[track_id].append(result)

    def get_consensus(self, track_id: int) -> Optional[ConsensusResult]:
        """
        Get majority-vote consensus across multiple frames.
        Returns the most voted plate text with confidence.
        """
        readings = self.readings.get(track_id, [])
        if len(readings) < 1:
            return None

        # Vote on plate text
        vote_counter = Counter()
        for r in readings:
            vote_counter[r.plate_text] += 1

        if not vote_counter:
            return None

        # Get winner
        plate_text, count = vote_counter.most_common(1)[0]
        confidence = count / len(readings)

        return ConsensusResult(
            plate_text=plate_text,
            confidence=confidence,
            num_frames=len(readings),
            votes=dict(vote_counter),
        )

    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[PlateResult]:
        """Process a full frame for plate detection."""
        plates = []
        candidates = self.detect_plate_region(frame)

        for bbox in candidates:
            result = self.read_plate(frame, bbox)
            if result:
                result.frame_id = frame_id
                plates.append(result)

        return plates

    def cleanup_old(self, max_age: int = 100):
        """Remove old readings to free memory."""
        to_remove = []
        for tid, readings in self.readings.items():
            if readings and (readings[-1].frame_id < max_age):
                to_remove.append(tid)
        for tid in to_remove:
            del self.readings[tid]
