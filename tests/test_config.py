"""Config — the model-swap flags that make a stock-vs-fine-tuned A/B possible.

ROADMAP §5.2 calls for swapping in fine-tuned weights behind a config flag
rather than a code edit. These tests pin that contract: the env vars are
read, the fine-tuned weights in models/weights/ are preferred when present,
and a clone without them still runs — on stock weights, loudly, rather than
failing. The defaults deliberately no longer stay stock: main.py and
web_demo.py used to resolve different models from the same repo.
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


def test_detection_defaults_to_the_fine_tuned_weights(monkeypatch):
    """The versioned checkpoint wins over stock without an env var set."""
    monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
    module = reload_config()
    assert module.DETECTION_WEIGHTS_DEFAULT.exists(), "weights must be tracked"
    assert module.DetectionConfig().model_path == str(module.DETECTION_WEIGHTS_DEFAULT)


def test_detection_falls_back_to_stock_when_weights_are_absent(monkeypatch, tmp_path):
    """A clone missing models/weights/ must still run, not crash."""
    monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
    module = reload_config()
    monkeypatch.setattr(module, "DETECTION_WEIGHTS_DEFAULT", tmp_path / "absent.pt")
    assert module.DetectionConfig().model_path == "yolov8n.pt"


def test_the_stock_fallback_warns(monkeypatch, tmp_path, caplog):
    """Silence is the bug: a demo on stock weights must say so."""
    monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
    module = reload_config()
    monkeypatch.setattr(module, "DETECTION_WEIGHTS_DEFAULT", tmp_path / "absent.pt")
    with caplog.at_level("WARNING", logger="ibvap.config"):
        module.DetectionConfig()
    assert "falling back" in caplog.text


def test_detection_model_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("IBVAP_DETECTION_MODEL", "runs/detect/ibvap_detection/weights/best.pt")
    module = reload_config()
    assert module.DetectionConfig().model_path.endswith("best.pt")
    assert module.CONFIG.detection.model_path.endswith("best.pt")


def test_plate_localizer_defaults_to_the_trained_weights(monkeypatch):
    """Contours score F1 0.194 against the trained localizer's 0.915."""
    monkeypatch.delenv("IBVAP_PLATE_MODEL", raising=False)
    module = reload_config()
    assert module.ANPRConfig().plate_model_path == str(module.PLATE_WEIGHTS_DEFAULT)


def test_plate_localizer_falls_back_to_contours_when_absent(monkeypatch, tmp_path):
    """None means the contour fallback — no model, no GPU, still runs."""
    monkeypatch.delenv("IBVAP_PLATE_MODEL", raising=False)
    module = reload_config()
    monkeypatch.setattr(module, "PLATE_WEIGHTS_DEFAULT", tmp_path / "absent.pt")
    assert module.ANPRConfig().plate_model_path is None


def test_plate_localizer_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("IBVAP_PLATE_MODEL", "runs/detect/ibvap_plate/weights/best.pt")
    assert reload_config().ANPRConfig().plate_model_path.endswith("best.pt")


def test_an_empty_env_var_forces_the_baseline(monkeypatch):
    """Set-but-empty is the A/B escape hatch, distinct from unset: it takes
    the baseline path even though the trained weights are sitting on disk."""
    monkeypatch.setenv("IBVAP_PLATE_MODEL", "")
    assert reload_config().ANPRConfig().plate_model_path is None


def test_an_empty_detection_env_var_forces_stock(monkeypatch):
    monkeypatch.setenv("IBVAP_DETECTION_MODEL", "")
    module = reload_config()
    assert module.DetectionConfig().model_path == "yolov8n.pt"
    assert module.DetectionConfig().confidence_threshold == 0.45


def test_detection_threshold_follows_the_model(monkeypatch, tmp_path):
    """0.25 is the fine-tuned model's F1 peak (scripts/evaluate.py threshold);
    0.45 was swept against stock COCO. Defaulting the weights without
    defaulting the cutoff would just relocate the mismatch."""
    monkeypatch.delenv("IBVAP_DETECTION_CONF", raising=False)
    monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
    module = reload_config()
    assert module.DetectionConfig().confidence_threshold == 0.25

    monkeypatch.setattr(module, "DETECTION_WEIGHTS_DEFAULT", tmp_path / "absent.pt")
    assert module.DetectionConfig().confidence_threshold == 0.45


def test_detection_threshold_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("IBVAP_DETECTION_CONF", "0.6")
    assert reload_config().DetectionConfig().confidence_threshold == 0.6


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
    assert loaded.detection.confidence_threshold == (
        config_module.DetectionConfig().confidence_threshold
    )
