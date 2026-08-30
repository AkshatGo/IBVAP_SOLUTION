"""Shared fixtures for the IBVAP test suite.

Nothing here loads YOLO or EasyOCR. The tests exercise the pipeline
*contract* — detection -> tracker -> fence -> ANPR -> hash chain — using
stub detections, so the suite runs in seconds on CPU with no model
downloads. Model quality is measured by scripts/evaluate.py, not here.
"""
import numpy as np
import pytest

from src.edge.detector import Detection


@pytest.fixture
def chain_path(tmp_path):
    """An isolated hash-chain file, so tests never touch data/hashchain.jsonl."""
    return str(tmp_path / "hashchain.jsonl")


@pytest.fixture
def frame():
    """A mid-grey 480x640 BGR frame — bright enough not to trip signal loss."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_detection(bbox, class_id=0, class_name="person", confidence=0.9, track_id=-1):
    """Build a Detection the way EdgeDetector would, with center derived."""
    x1, y1, x2, y2 = bbox
    return Detection(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center=((x1 + x2) // 2, (y1 + y2) // 2),
    )


@pytest.fixture
def detection_factory():
    return make_detection
