"""
IBVAP Edge Detection Module - Unit Tests
Comprehensive tests for EdgeDetector, ANPREngine, SignalLossDetector, and HashChainVerifier
"""

import pytest
import numpy as np
import time
import hashlib
import json
from typing import Dict, List
from unittest.mock import MagicMock, patch

from src.edge.detector import (
    EdgeDetector,
    ANPREngine,
    SignalLossDetector,
    HashChainVerifier,
    Detection,
    Track,
    VirtualFence,
)


# ============================================================
# EdgeDetector Tests
# ============================================================

class TestEdgeDetector:
    """Tests for the main EdgeDetector class"""

    def test_initialization(self):
        """Test EdgeDetector initializes correctly"""
        detector = EdgeDetector(model_name="yolov8n", confidence_threshold=0.5)
        
        assert detector.model_name == "yolov8n"
        assert detector.confidence_threshold == 0.5
        assert detector.tracks == {}
        assert detector.virtual_fences == []
        assert detector.alert_history == []
        assert detector.frame_count == 0
        assert detector.prev_hash == "0" * 64

    def test_initialization_default_values(self):
        """Test EdgeDetector with default parameters"""
        detector = EdgeDetector()
        
        assert detector.model_name == "yolov8n"
        assert detector.confidence_threshold == 0.5

    def test_add_virtual_fence(self, simple_fence):
        """Test adding a virtual fence"""
        detector = EdgeDetector()
        detector.add_virtual_fence(simple_fence)
        
        assert len(detector.virtual_fences) == 1
        assert detector.virtual_fences[0].fence_id == "fence-001"
        assert detector.virtual_fences[0].zone_name == "Zone-1"

    def test_add_multiple_virtual_fences(self, simple_fence, complex_fence):
        """Test adding multiple virtual fences"""
        detector = EdgeDetector()
        detector.add_virtual_fence(simple_fence)
        detector.add_virtual_fence(complex_fence)
        
        assert len(detector.virtual_fences) == 2
        assert detector.virtual_fences[0].zone_name == "Zone-1"
        assert detector.virtual_fences[1].zone_name == "Zone-2"

    def test_process_frame_basic(self, sample_frame):
        """Test basic frame processing"""
        detector = EdgeDetector()
        timestamp = time.time()
        
        result = detector.process_frame(sample_frame, timestamp)
        
        assert 'frame_idx' in result
        assert 'timestamp' in result
        assert 'detections' in result
        assert 'alerts' in result
        assert 'active_tracks' in result
        assert 'detection_time_ms' in result
        assert 'fps' in result
        
        assert result['frame_idx'] == 1
        assert result['timestamp'] == timestamp
        assert isinstance(result['detections'], list)
        assert isinstance(result['alerts'], list)

    def test_process_frame_increments_counter(self, sample_frame):
        """Test that frame counter increments"""
        detector = EdgeDetector()
        
        for i in range(5):
            detector.process_frame(sample_frame, time.time())
        
        assert detector.frame_count == 5

    def test_process_frame_empty(self, empty_frame):
        """Test processing empty frame (no objects)"""
        detector = EdgeDetector()
        timestamp = time.time()
        
        result = detector.process_frame(empty_frame, timestamp)
        
        assert result['frame_idx'] == 1
        assert result['active_tracks'] == 0

    def test_performance_stats(self, sample_frame):
        """Test performance statistics"""
        detector = EdgeDetector()
        
        # Process some frames
        for _ in range(10):
            detector.process_frame(sample_frame, time.time())
        
        stats = detector.get_performance_stats()
        
        assert stats['total_frames'] == 10
        assert stats['total_active_tracks'] >= 0
        assert stats['total_alerts'] >= 0
        assert stats['avg_detection_time_ms'] >= 0
        assert stats['avg_fps'] > 0

    def test_point_in_polygon_inside(self):
        """Test point inside polygon detection"""
        detector = EdgeDetector()
        polygon = [(100, 100), (500, 100), (500, 400), (100, 400)]
        
        # Point clearly inside
        assert detector._point_in_polygon((250, 250), polygon) is True

    def test_point_in_polygon_outside(self):
        """Test point outside polygon detection"""
        detector = EdgeDetector()
        polygon = [(100, 100), (500, 100), (500, 400), (100, 400)]
        
        # Point clearly outside
        assert detector._point_in_polygon((50, 50), polygon) is False
        assert detector._point_in_polygon((600, 600), polygon) is False

    def test_point_in_polygon_boundary(self):
        """Test point on polygon boundary"""
        detector = EdgeDetector()
        polygon = [(100, 100), (500, 100), (500, 400), (100, 400)]
        
        # Point on boundary (edge case)
        result = detector._point_in_polygon((100, 250), polygon)
        # Result depends on implementation, just check it doesn't crash
        assert isinstance(result, bool)

    def test_generate_explanation(self, sample_detection_data, simple_fence):
        """Test explanation generation"""
        detector = EdgeDetector()
        
        detection = Detection(
            track_id=42,
            class_name=sample_detection_data['class_name'],
            confidence=sample_detection_data['confidence'],
            bbox=sample_detection_data['bbox'],
            centroid=sample_detection_data['centroid'],
            timestamp=time.time(),
            frame_idx=1
        )
        
        explanation = detector._generate_explanation(detection, simple_fence)
        
        assert isinstance(explanation, str)
        assert "T-0042" in explanation
        assert "Zone-1" in explanation
        assert "person" in explanation

    def test_calculate_bearing_no_history(self):
        """Test bearing calculation with no track history"""
        detector = EdgeDetector()
        
        detection = Detection(
            track_id=99,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            timestamp=time.time(),
            frame_idx=1
        )
        
        bearing = detector._calculate_bearing(detection)
        assert bearing == "N/A"

    def test_calculate_bearing_with_history(self):
        """Test bearing calculation with track history"""
        detector = EdgeDetector()
        
        # Add track history
        track = Track(track_id=1, class_name="person")
        track.detections = [
            Detection(1, "person", 0.9, (100, 100, 200, 200), (150, 150), time.time() - 0.1, 1),
            Detection(1, "person", 0.9, (110, 110, 210, 210), (160, 160), time.time(), 2),
        ]
        detector.tracks[1] = track
        
        detection = track.detections[-1]
        bearing = detector._calculate_bearing(detection)
        
        assert isinstance(bearing, str)
        assert bearing in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    def test_create_alert(self, simple_fence):
        """Test alert creation with hash chain"""
        detector = EdgeDetector()
        
        detection = Detection(
            track_id=1,
            class_name="person",
            confidence=0.91,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            timestamp=time.time(),
            frame_idx=1
        )
        
        alert = detector._create_alert(
            event_type="fence_intrusion",
            detection=detection,
            fence=simple_fence,
            explanation="Test alert",
            timestamp=time.time()
        )
        
        assert 'event_id' in alert
        assert 'prev_hash' in alert
        assert alert['prev_hash'] == "0" * 64  # Genesis hash
        assert alert['site_id'] == "BOP-01"
        assert alert['camera_id'] == "CAM-01"
        assert alert['event_type'] == "fence_intrusion"
        assert alert['severity'] == "high"

    def test_hash_chain_updates(self, simple_fence):
        """Test that hash chain updates correctly"""
        detector = EdgeDetector()
        initial_hash = detector.prev_hash
        
        detection = Detection(
            track_id=1,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            timestamp=time.time(),
            frame_idx=1
        )
        
        detector._create_alert(
            event_type="fence_intrusion",
            detection=detection,
            fence=simple_fence,
            explanation="Test",
            timestamp=time.time()
        )
        
        assert detector.prev_hash != initial_hash
        assert len(detector.prev_hash) == 64

    def test_multiple_alerts_chain(self, simple_fence):
        """Test hash chain with multiple alerts"""
        detector = EdgeDetector()
        alerts = []
        
        for i in range(5):
            detection = Detection(
                track_id=i,
                class_name="person",
                confidence=0.9,
                bbox=(100, 100, 200, 200),
                centroid=(150, 150),
                timestamp=time.time(),
                frame_idx=i
            )
            
            alert = detector._create_alert(
                event_type="fence_intrusion",
                detection=detection,
                fence=simple_fence,
                explanation=f"Alert {i}",
                timestamp=time.time()
            )
            alerts.append(alert)
        
        # Verify chain
        is_valid, _ = HashChainVerifier.verify_chain(alerts)
        assert is_valid is True

    def test_assign_track_id_new(self):
        """Test track ID assignment for new object"""
        detector = EdgeDetector()
        
        track_id = detector._assign_track_id((150, 250), "person")
        
        assert isinstance(track_id, int)
        assert track_id > 0

    def test_assign_track_id_existing(self):
        """Test track ID assignment for existing object"""
        detector = EdgeDetector()
        
        # Add existing track
        track = Track(track_id=1, class_name="person")
        track.detections = [
            Detection(1, "person", 0.9, (100, 100, 200, 200), (150, 150), time.time(), 1)
        ]
        detector.tracks[1] = track
        
        # Assign ID for nearby point
        track_id = detector._assign_track_id((155, 155), "person")
        
        assert track_id == 1  # Should reuse existing track


# ============================================================
# Detection Dataclass Tests
# ============================================================

class TestDetection:
    """Tests for Detection dataclass"""

    def test_detection_creation(self):
        """Test creating a Detection object"""
        detection = Detection(
            track_id=1,
            class_name="person",
            confidence=0.95,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            timestamp=time.time(),
            frame_idx=1
        )
        
        assert detection.track_id == 1
        assert detection.class_name == "person"
        assert detection.confidence == 0.95
        assert detection.bbox == (100, 100, 200, 200)
        assert detection.centroid == (150, 150)

    def test_detection_types(self):
        """Test Detection field types"""
        detection = Detection(
            track_id=1,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            timestamp=time.time(),
            frame_idx=1
        )
        
        assert isinstance(detection.track_id, int)
        assert isinstance(detection.class_name, str)
        assert isinstance(detection.confidence, float)
        assert isinstance(detection.bbox, tuple)
        assert isinstance(detection.centroid, tuple)


# ============================================================
# Track Dataclass Tests
# ============================================================

class TestTrack:
    """Tests for Track dataclass"""

    def test_track_creation(self):
        """Test creating a Track object"""
        track = Track(track_id=1, class_name="person")
        
        assert track.track_id == 1
        assert track.class_name == "person"
        assert track.detections == []
        assert track.is_active is True

    def test_track_age_no_detections(self):
        """Test track age with no detections"""
        track = Track(track_id=1, class_name="person")
        
        assert track.age == 0.0

    def test_track_age_with_detections(self):
        """Test track age calculation"""
        track = Track(track_id=1, class_name="person")
        track.first_seen = time.time() - 1.0
        track.last_seen = time.time()
        
        age = track.age
        
        assert 0.9 <= age <= 1.1  # Allow small time difference

    def test_track_velocity_no_detections(self):
        """Test velocity with no detections"""
        track = Track(track_id=1, class_name="person")
        
        velocity = track.velocity
        
        assert velocity == (0.0, 0.0)

    def test_track_velocity_single_detection(self):
        """Test velocity with single detection"""
        track = Track(track_id=1, class_name="person")
        track.detections = [
            Detection(1, "person", 0.9, (100, 100, 200, 200), (150, 150), time.time(), 1)
        ]
        
        velocity = track.velocity
        
        assert velocity == (0.0, 0.0)

    def test_track_velocity_multiple_detections(self):
        """Test velocity calculation with multiple detections"""
        track = Track(track_id=1, class_name="person")
        t1 = time.time()
        t2 = t1 + 0.5
        
        track.detections = [
            Detection(1, "person", 0.9, (100, 100, 200, 200), (100, 100), t1, 1),
            Detection(1, "person", 0.9, (150, 150, 250, 250), (150, 150), t2, 2),
        ]
        
        velocity = track.velocity
        
        # Should be moving right and down
        assert velocity[0] > 0  # x velocity
        assert velocity[1] > 0  # y velocity


# ============================================================
# VirtualFence Tests
# ============================================================

class TestVirtualFence:
    """Tests for VirtualFence dataclass"""

    def test_fence_creation(self, simple_fence):
        """Test creating a VirtualFence object"""
        assert simple_fence.fence_id == "fence-001"
        assert simple_fence.zone_name == "Zone-1"
        assert len(simple_fence.polygon) == 4
        assert simple_fence.severity == "high"
        assert simple_fence.is_active is True

    def test_fence_inactive(self):
        """Test inactive fence"""
        fence = VirtualFence(
            fence_id="fence-inactive",
            zone_name="Zone-X",
            polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            severity="low",
            is_active=False
        )
        
        assert fence.is_active is False


# ============================================================
# ANPREngine Tests
# ============================================================

class TestANPREngine:
    """Tests for ANPREngine class"""

    def test_initialization(self):
        """Test ANPREngine initializes correctly"""
        anpr = ANPREngine(ocr_model="paddleocr")
        
        assert anpr.ocr_model == "paddleocr"
        assert anpr.consensus_threshold == 0.6
        assert anpr.frame_results == {}

    def test_extract_plate_region(self, sample_frame, sample_bbox):
        """Test plate region extraction"""
        anpr = ANPREngine()
        
        plate_region = anpr._extract_plate_region(sample_frame, sample_bbox)
        
        assert isinstance(plate_region, np.ndarray)
        assert plate_region.ndim == 3  # Should be RGB image

    def test_run_ocr_success(self):
        """Test successful OCR"""
        anpr = ANPREngine()
        plate_region = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        
        # Mock random to always succeed
        with patch('random.random', return_value=0.8):
            result = anpr._run_ocr(plate_region)
        
        assert result is not None
        assert isinstance(result, str)

    def test_run_ocr_failure(self):
        """Test OCR failure"""
        anpr = ANPREngine()
        plate_region = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
        
        # Mock random to always fail
        with patch('random.random', return_value=0.1):
            result = anpr._run_ocr(plate_region)
        
        assert result is None

    def test_run_ocr_empty_region(self):
        """Test OCR with empty region"""
        anpr = ANPREngine()
        plate_region = np.array([])
        
        result = anpr._run_ocr(plate_region)
        
        assert result is None

    def test_get_consensus_insufficient_frames(self):
        """Test consensus with insufficient frames"""
        anpr = ANPREngine()
        anpr.frame_results[1] = ["BR12AB3456"]
        
        consensus = anpr._get_consensus(1)
        
        assert consensus is None  # Need at least 3 frames

    def test_get_consensus_no_agreement(self):
        """Test consensus with no agreement"""
        anpr = ANPREngine()
        anpr.frame_results[1] = ["AAA", "BBB", "CCC", "DDD"]
        
        consensus = anpr._get_consensus(1)
        
        assert consensus is None  # No majority

    def test_get_consensus_strong_agreement(self):
        """Test consensus with strong agreement"""
        anpr = ANPREngine()
        anpr.frame_results[1] = ["BR12AB3456", "BR12AB3456", "BR12AB3456", "BR12AB3458"]
        
        consensus = anpr._get_consensus(1)
        
        assert consensus == "BR12AB3456"  # 3/4 agreement

    def test_get_stats(self):
        """Test statistics calculation"""
        anpr = ANPREngine()
        anpr.frame_results[1] = ["BR12AB3456", "BR12AB3456", "BR12AB3456"]
        anpr.frame_results[2] = ["DL01CD7890"]
        
        stats = anpr.get_stats()
        
        assert stats['total_frames_processed'] == 4
        assert stats['tracks_with_consensus'] == 1
        assert stats['consensus_rate'] == 0.5


# ============================================================
# SignalLossDetector Tests
# ============================================================

class TestSignalLossDetector:
    """Tests for SignalLossDetector class"""

    def test_initialization(self):
        """Test SignalLossDetector initializes correctly"""
        detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=5.0)
        
        assert detector.camera_id == "CAM-01"
        assert detector.timeout_seconds == 5.0
        assert detector.is_online is True
        assert detector.last_frame_time == 0.0

    def test_update_returns_none(self):
        """Test update returns None when online"""
        detector = SignalLossDetector(camera_id="CAM-01")
        
        result = detector.update(time.time())
        
        assert result is None
        assert detector.is_online is True

    def test_check_timeout_no_timeout(self):
        """Test check_timeout when not timed out"""
        detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=5.0)
        detector.update(time.time())
        
        result = detector.check_timeout(time.time())
        
        assert result is None
        assert detector.is_online is True

    def test_check_timeout_with_timeout(self):
        """Test check_timeout when timed out"""
        detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=1.0)
        detector.update(time.time() - 2.0)  # 2 seconds ago
        
        result = detector.check_timeout(time.time())
        
        assert result is not None
        assert result['event_type'] == 'signal_loss'
        assert result['severity'] == 'critical'
        assert result['camera_id'] == 'CAM-01'
        assert detector.is_online is False

    def test_signal_loss_alert_format(self):
        """Test signal loss alert has correct format"""
        detector = SignalLossDetector(camera_id="CAM-02", timeout_seconds=1.0)
        detector.update(time.time() - 2.0)
        
        alert = detector.check_timeout(time.time())
        
        assert 'event_id' in alert
        assert 'prev_hash' in alert
        assert 'timestamp' in alert
        assert alert['camera_id'] == 'CAM-02'
        assert alert['object_class'] == 'none'
        assert alert['track_id'] == 'N/A'

    def test_hash_chain_updates_on_signal_loss(self):
        """Test hash chain updates on signal loss"""
        detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=1.0)
        detector.update(time.time() - 2.0)
        
        initial_hash = detector.prev_hash
        detector.check_timeout(time.time())
        
        assert detector.prev_hash != initial_hash
        assert len(detector.prev_hash) == 64


# ============================================================
# HashChainVerifier Tests
# ============================================================

class TestHashChainVerifier:
    """Tests for HashChainVerifier class"""

    def test_verify_empty_chain(self):
        """Test verifying empty chain"""
        is_valid, tampered_idx = HashChainVerifier.verify_chain([])
        
        assert is_valid is True
        assert tampered_idx is None

    def test_verify_single_event(self):
        """Test verifying single event chain"""
        events = [
            {
                'event_id': 'e001',
                'prev_hash': '0' * 64,
                'data': 'test'
            }
        ]
        
        is_valid, tampered_idx = HashChainVerifier.verify_chain(events)
        
        assert is_valid is True
        assert tampered_idx is None

    def test_verify_valid_chain(self):
        """Test verifying valid hash chain"""
        events = []
        prev_hash = "0" * 64
        
        for i in range(5):
            event = {
                'event_id': f'e{i:03d}',
                'prev_hash': prev_hash,
                'data': f'event_{i}'
            }
            
            # Calculate hash for next event
            event_for_hash = event.copy()
            event_for_hash.pop('prev_hash')
            prev_hash = hashlib.sha256(
                json.dumps(event_for_hash, sort_keys=True).encode()
            ).hexdigest()
            
            events.append(event)
        
        is_valid, tampered_idx = HashChainVerifier.verify_chain(events)
        
        assert is_valid is True
        assert tampered_idx is None

    def test_verify_tampered_chain(self):
        """Test detecting tampered hash chain"""
        events = []
        prev_hash = "0" * 64
        
        for i in range(5):
            event = {
                'event_id': f'e{i:03d}',
                'prev_hash': prev_hash,
                'data': f'event_{i}'
            }
            
            # Calculate hash for next event
            event_for_hash = event.copy()
            event_for_hash.pop('prev_hash')
            prev_hash = hashlib.sha256(
                json.dumps(event_for_hash, sort_keys=True).encode()
            ).hexdigest()
            
            events.append(event)
        
        # Tamper with event 2
        events[2]['data'] = 'TAMPERED'
        
        is_valid, tampered_idx = HashChainVerifier.verify_chain(events)
        
        assert is_valid is False
        assert tampered_idx == 3  # Tamper detected at next event


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for the full pipeline"""

    def test_full_pipeline_single_frame(self, sample_frame):
        """Test complete pipeline with single frame"""
        detector = EdgeDetector()
        fence = VirtualFence(
            fence_id="fence-001",
            zone_name="Zone-1",
            polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
            severity="high"
        )
        detector.add_virtual_fence(fence)
        
        timestamp = time.time()
        result = detector.process_frame(sample_frame, timestamp)
        
        assert result['frame_idx'] == 1
        assert isinstance(result['detections'], list)
        assert isinstance(result['alerts'], list)

    def test_full_pipeline_multiple_frames(self, sample_frame):
        """Test complete pipeline with multiple frames"""
        detector = EdgeDetector()
        fence = VirtualFence(
            fence_id="fence-001",
            zone_name="Zone-1",
            polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
            severity="high"
        )
        detector.add_virtual_fence(fence)
        
        # Process 50 frames
        for i in range(50):
            timestamp = time.time() + i * 0.033
            result = detector.process_frame(sample_frame, timestamp)
        
        stats = detector.get_performance_stats()
        
        assert stats['total_frames'] == 50
        assert stats['avg_fps'] > 0

    def test_anpr_with_detector(self, sample_frame, sample_bbox):
        """Test ANPR integrated with detector"""
        detector = EdgeDetector()
        anpr = ANPREngine()
        
        # Process frame
        result = detector.process_frame(sample_frame, time.time())
        
        # Run ANPR on any detections
        for det in result['detections']:
            plate = anpr.process_frame(sample_frame, det.track_id, det.bbox)
            # Plate may or may not be found, just test no crash
        
        assert True  # No exception

    def test_signal_loss_with_detector(self, sample_frame):
        """Test signal loss detection with main detector"""
        detector = EdgeDetector()
        signal_detector = SignalLossDetector(camera_id="CAM-01", timeout_seconds=1.0)
        
        # First frame - online
        signal_detector.update(time.time())
        detector.process_frame(sample_frame, time.time())
        
        # Wait for timeout
        time.sleep(1.5)
        
        # Check for signal loss
        alert = signal_detector.check_timeout(time.time())
        
        assert alert is not None
        assert alert['event_type'] == 'signal_loss'

    def test_end_to_end_alert_chain(self, sample_frame):
        """Test end-to-end alert chain integrity"""
        detector = EdgeDetector()
        fence = VirtualFence(
            fence_id="fence-001",
            zone_name="Zone-1",
            polygon=[(0, 0), (640, 0), (640, 480), (0, 480)],
            severity="high"
        )
        detector.add_virtual_fence(fence)
        
        # Process frames and collect alerts
        all_alerts = []
        for i in range(10):
            result = detector.process_frame(sample_frame, time.time())
            all_alerts.extend(result['alerts'])
        
        # Verify hash chain if any alerts
        if all_alerts:
            is_valid, _ = HashChainVerifier.verify_chain(all_alerts)
            assert is_valid is True


# ============================================================
# Performance Tests
# ============================================================

class TestPerformance:
    """Performance tests for edge detection"""

    def test_processing_speed(self, sample_frame):
        """Test that processing is fast enough for real-time"""
        detector = EdgeDetector()
        
        start_time = time.time()
        num_frames = 100
        
        for _ in range(num_frames):
            detector.process_frame(sample_frame, time.time())
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        # Should process at least 10 FPS (very conservative)
        assert fps >= 10, f"Processing too slow: {fps:.1f} FPS"

    def test_memory_usage(self, sample_frame):
        """Test that memory doesn't grow excessively"""
        detector = EdgeDetector()
        
        # Process many frames
        for _ in range(200):
            detector.process_frame(sample_frame, time.time())
        
        # Check that tracks are being managed
        stats = detector.get_performance_stats()
        
        # Should not have excessive tracks
        assert stats['total_active_tracks'] < 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
