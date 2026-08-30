"""Signal-loss detector — tamper heuristics and config-driven thresholds.

A blinded camera is itself an alert, so the tests cover each way a feed can
go bad: covered lens (black), sensor failure (white), gradual interference
(brightness drop) and a dead link (timeout).
"""
import json

import numpy as np
import pytest

from src.edge import signal as signal_module
from src.edge.signal import SignalLossDetector

CAM = "CAM-TEST"


def frame_at(brightness, shape=(64, 64, 3)):
    return np.full(shape, brightness, dtype=np.uint8)


@pytest.fixture
def detector(monkeypatch, tmp_path):
    """A detector with known thresholds, isolated from config/ on disk."""
    monkeypatch.setattr(signal_module, "_THRESHOLDS_PATH", str(tmp_path / "absent.json"))
    det = SignalLossDetector(
        timeout_seconds=5.0, black_threshold=5.0, white_threshold=250.0,
        brightness_drop_threshold=0.7, consecutive_black_needed=3,
    )
    det.register_camera(CAM)
    return det


def test_a_normal_frame_raises_nothing(detector):
    assert detector.update(CAM, frame_at(128)) is None


def test_black_frames_alert_only_after_the_consecutive_threshold(detector):
    """One dark frame is noise; three in a row is a covered lens."""
    assert detector.update(CAM, frame_at(0)) is None
    assert detector.update(CAM, frame_at(0)) is None
    message = detector.update(CAM, frame_at(0))
    assert message is not None and "black" in message.lower()


def test_the_black_frame_counter_resets_on_a_good_frame(detector):
    detector.update(CAM, frame_at(0))
    detector.update(CAM, frame_at(0))
    detector.update(CAM, frame_at(128))   # recovery resets the run
    assert detector.update(CAM, frame_at(0)) is None


def test_a_white_frame_alerts_immediately(detector):
    message = detector.update(CAM, frame_at(255))
    assert message is not None and "white" in message.lower()


def test_an_empty_frame_alerts(detector):
    message = detector.update(CAM, np.array([], dtype=np.uint8))
    assert message is not None


def test_a_sudden_brightness_drop_alerts(detector):
    """Regression: this compared the new reading against itself and never fired."""
    for _ in range(12):                    # get past the frame_count > 10 gate
        detector.update(CAM, frame_at(200))
    message = detector.update(CAM, frame_at(60))   # 60/200 = 0.30 < 0.70
    assert message is not None and "brightness" in message.lower()


def test_a_gentle_brightness_change_does_not_alert(detector):
    for _ in range(12):
        detector.update(CAM, frame_at(200))
    assert detector.update(CAM, frame_at(180)) is None   # 0.90 > 0.70


def test_camera_goes_offline_then_reports_restoration(detector):
    for _ in range(3):
        detector.update(CAM, frame_at(0))
    assert detector.get_offline_count() == 1

    message = detector.update(CAM, frame_at(128))
    assert message is not None and "restored" in message.lower()
    assert detector.get_online_count() == 1


def test_status_reports_every_registered_camera(detector):
    detector.update(CAM, frame_at(128))
    status = detector.get_status()
    assert CAM in status
    assert status[CAM]["is_online"] is True


def test_an_unregistered_camera_is_registered_on_first_frame(detector):
    detector.update("CAM-NEW", frame_at(128))
    assert "CAM-NEW" in detector.get_status()


def test_thresholds_are_read_from_the_config_file(monkeypatch, tmp_path):
    """The whole point of the config file: tune sensitivity without a redeploy."""
    config_path = tmp_path / "signal_thresholds.json"
    config_path.write_text(json.dumps({
        "black_threshold": 90.0,          # deliberately extreme
        "consecutive_black_needed": 1,
    }))
    monkeypatch.setattr(signal_module, "_THRESHOLDS_PATH", str(config_path))

    detector = SignalLossDetector()
    assert detector.black_threshold == 90.0
    assert detector.consecutive_black_needed == 1

    # A frame that would be "normal" under the defaults now trips the alarm.
    message = detector.update(CAM, frame_at(50))
    assert message is not None and "black" in message.lower()


def test_thresholds_hot_reload(monkeypatch, tmp_path):
    config_path = tmp_path / "signal_thresholds.json"
    config_path.write_text(json.dumps({"black_threshold": 5.0}))
    monkeypatch.setattr(signal_module, "_THRESHOLDS_PATH", str(config_path))

    detector = SignalLossDetector()
    assert detector.black_threshold == 5.0

    config_path.write_text(json.dumps({"black_threshold": 42.0}))
    detector.reload_thresholds()
    assert detector.black_threshold == 42.0
    assert detector.get_thresholds()["black_threshold"] == 42.0


def test_a_malformed_config_falls_back_to_defaults(monkeypatch, tmp_path):
    config_path = tmp_path / "signal_thresholds.json"
    config_path.write_text("{ not valid json")
    monkeypatch.setattr(signal_module, "_THRESHOLDS_PATH", str(config_path))

    detector = SignalLossDetector(black_threshold=5.0)
    assert detector.black_threshold == 5.0


def test_timeout_marks_a_silent_camera_offline(detector):
    detector.timeout_seconds = 0.0
    detector.update(CAM, frame_at(128))
    assert detector.check_timeout(CAM) is not None
