"""Config — the model-swap flags that make a stock-vs-fine-tuned A/B possible.

ROADMAP §5.2 calls for swapping in fine-tuned weights behind a config flag
rather than a code edit. These tests pin that contract: the env vars are
read, and the defaults stay stock so a fresh clone runs without any
training artifacts present.
"""
import importlib

import pytest

import src.config as config_module


def reload_config():
    """Re-import the config module so env vars are re-read."""
    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def restore_config():
    """Reload after each test so the module singleton doesn't leak env state.

    monkeypatch restores the env vars themselves; this re-reads them into
    CONFIG, which was built at import time.
    """
    yield
    reload_config()


def test_detection_defaults_to_stock_yolov8n(monkeypatch):
    monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
    assert reload_config().DetectionConfig().model_path == "yolov8n.pt"


def test_detection_model_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("IBVAP_DETECTION_MODEL", "runs/detect/ibvap_detection/weights/best.pt")
    module = reload_config()
    assert module.DetectionConfig().model_path.endswith("best.pt")
    assert module.CONFIG.detection.model_path.endswith("best.pt")


def test_plate_localizer_defaults_to_none(monkeypatch):
    """None means the contour fallback — a fresh clone needs no checkpoint."""
    monkeypatch.delenv("IBVAP_PLATE_MODEL", raising=False)
    assert reload_config().ANPRConfig().plate_model_path is None


def test_plate_localizer_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("IBVAP_PLATE_MODEL", "runs/detect/ibvap_plate/weights/best.pt")
    assert reload_config().ANPRConfig().plate_model_path.endswith("best.pt")


def test_an_empty_env_var_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("IBVAP_PLATE_MODEL", "")
    assert reload_config().ANPRConfig().plate_model_path is None


def test_detection_threshold_is_the_documented_value():
    """0.45 is referenced by scripts/evaluate.py threshold; keep them in step."""
    assert config_module.DetectionConfig().confidence_threshold == 0.45


def test_target_classes_match_the_six_trained_classes():
    """COCO indices for the classes the dataset scripts keep."""
    assert config_module.DetectionConfig().target_classes == [0, 1, 2, 3, 5, 7]


def test_config_round_trips_through_disk(tmp_path):
    path = tmp_path / "ibvap_config.json"
    config_module.IBVAPConfig().save(str(path))
    assert path.exists()
    assert config_module.IBVAPConfig.load(str(path)) is not None


def test_load_returns_defaults_when_no_file_exists(tmp_path):
    loaded = config_module.IBVAPConfig.load(str(tmp_path / "absent.json"))
    assert loaded.detection.confidence_threshold == 0.45
