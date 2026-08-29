"""
IBVAP Test Configuration and Fixtures
Provides shared test utilities and fixtures for all test modules
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch


# ============================================================
# Test Data Fixtures
# ============================================================

@pytest.fixture
def sample_frame():
    """Create a sample RGB frame for testing"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_small():
    """Create a small RGB frame for fast testing"""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def empty_frame():
    """Create a black frame (no objects)"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Sample bounding box (x1, y1, x2, y2)"""
    return (100, 150, 200, 350)


@pytest.fixture
def sample_centroid():
    """Sample centroid coordinates"""
    return (150, 250)


@pytest.fixture
def sample_polygon():
    """Sample virtual fence polygon"""
    return [(100, 100), (500, 100), (500, 400), (100, 400)]


@pytest.fixture
def sample_polygon_complex():
    """More complex polygon for edge case testing"""
    return [(200, 100), (400, 50), (500, 200), (350, 400), (150, 350)]


@pytest.fixture
def sample_detection_data():
    """Sample detection dictionary"""
    return {
        "class_name": "person",
        "confidence": 0.87,
        "bbox": (100, 150, 200, 350),
        "centroid": (150, 250)
    }


@pytest.fixture
def sample_vehicle_detection():
    """Sample vehicle detection"""
    return {
        "class_name": "vehicle",
        "confidence": 0.92,
        "bbox": (300, 200, 500, 350),
        "centroid": (400, 275)
    }


# ============================================================
# Virtual Fence Fixtures
# ============================================================

@pytest.fixture
def simple_fence():
    """Simple rectangular virtual fence"""
    from src.edge.detector import VirtualFence
    return VirtualFence(
        fence_id="fence-001",
        zone_name="Zone-1",
        polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
        severity="high"
    )


@pytest.fixture
def complex_fence():
    """Complex polygon virtual fence"""
    from src.edge.detector import VirtualFence
    return VirtualFence(
        fence_id="fence-002",
        zone_name="Zone-2",
        polygon=[(200, 100), (400, 50), (500, 200), (350, 400), (150, 350)],
        severity="critical"
    )


@pytest.fixture
def low_severity_fence():
    """Low severity fence for testing"""
    from src.edge.detector import VirtualFence
    return VirtualFence(
        fence_id="fence-003",
        zone_name="Zone-3",
        polygon=[(50, 50), (550, 50), (550, 450), (50, 450)],
        severity="low"
    )


# ============================================================
# Time Fixtures
# ============================================================

@pytest.fixture
def current_time():
    """Current timestamp"""
    return time.time()


@pytest.fixture
def timestamp_sequence():
    """Sequence of timestamps for multi-frame testing"""
    base_time = time.time()
    return [base_time + i * 0.033 for i in range(100)]


# ============================================================
# Mock Fixtures
# ============================================================

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing"""
    mock = MagicMock()
    mock.predict.return_value = [MagicMock()]
    return mock


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine for testing"""
    mock = MagicMock()
    mock.ocr.return_value = [[["BR12AB3456", 0.95]]]
    return mock


# ============================================================
# Helper Functions
# ============================================================

def create_test_frame_with_object(
    obj_class: str = "person",
    bbox: Tuple[int, int, int, int] = (100, 150, 200, 350),
    frame_size: Tuple[int, int] = (480, 640)
) -> np.ndarray:
    """Create a test frame with a colored rectangle representing an object"""
    frame = np.random.randint(0, 50, (*frame_size, 3), dtype=np.uint8)  # Dark background
    
    x1, y1, x2, y2 = bbox
    
    # Draw a colored rectangle for the object
    if obj_class == "person":
        color = (0, 255, 0)  # Green for person
    elif obj_class == "vehicle":
        color = (255, 0, 0)  # Red for vehicle
    else:
        color = (0, 0, 255)  # Blue for other
    
    frame[y1:y2, x1:x2] = color
    
    return frame


def create_track_history(
    track_id: int = 1,
    num_frames: int = 10,
    start_pos: Tuple[int, int] = (100, 200),
    velocity: Tuple[int, int] = (5, 2)
) -> List[Tuple[float, Tuple[int, int]]]:
    """Create a track history with movement"""
    base_time = time.time()
    history = []
    
    for i in range(num_frames):
        x = start_pos[0] + velocity[0] * i
        y = start_pos[1] + velocity[1] * i
        timestamp = base_time + i * 0.033
        history.append((timestamp, (x, y)))
    
    return history


# ============================================================
# Assertion Helpers
# ============================================================

def assert_detection_valid(detection):
    """Assert that a detection object has valid fields"""
    assert hasattr(detection, 'track_id')
    assert hasattr(detection, 'class_name')
    assert hasattr(detection, 'confidence')
    assert hasattr(detection, 'bbox')
    assert hasattr(detection, 'centroid')
    assert hasattr(detection, 'timestamp')
    assert hasattr(detection, 'frame_idx')
    
    assert isinstance(detection.track_id, int)
    assert isinstance(detection.class_name, str)
    assert isinstance(detection.confidence, float)
    assert 0.0 <= detection.confidence <= 1.0
    assert isinstance(detection.bbox, tuple)
    assert len(detection.bbox) == 4
    assert isinstance(detection.centroid, tuple)
    assert len(detection.centroid) == 2


def assert_alert_valid(alert: Dict):
    """Assert that an alert dictionary has valid fields"""
    required_fields = [
        'event_id', 'prev_hash', 'timestamp', 'site_id', 'camera_id',
        'event_type', 'object_class', 'track_id', 'zone', 'confidence',
        'explanation', 'severity', 'clip_ref'
    ]
    
    for field in required_fields:
        assert field in alert, f"Missing required field: {field}"
    
    assert isinstance(alert['confidence'], float)
    assert 0.0 <= alert['confidence'] <= 1.0
    assert alert['severity'] in ['low', 'medium', 'high', 'critical']
    assert isinstance(alert['prev_hash'], str)
    assert len(alert['prev_hash']) == 64  # SHA-256 hash


def assert_hash_chain_valid(events: List[Dict]):
    """Assert that a list of events forms a valid hash chain"""
    import hashlib
    import json
    
    for i in range(1, len(events)):
        prev_event = events[i - 1].copy()
        prev_event.pop('prev_hash', None)
        
        calculated_hash = hashlib.sha256(
            json.dumps(prev_event, sort_keys=True).encode()
        ).hexdigest()
        
        assert calculated_hash == events[i]['prev_hash'], \
            f"Hash chain broken at index {i}"
