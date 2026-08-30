"""ANPR — consensus voting and localizer selection.

OCR itself is not tested here: EasyOCR is a pretrained third-party model,
and asserting on its output would be testing their model, not our code.
What is ours is the multi-frame majority vote and the choice of
localization path, so that is what these tests cover. End-to-end plate
accuracy is measured by scripts/evaluate.py against a labelled set.
"""
import sys

import pytest

from src.edge.anpr import ANPREngine, PlateResult


@pytest.fixture
def no_easyocr(monkeypatch):
    """Make `import easyocr` fail, so load() never pulls OCR weights.

    Without this the test downloads ~100MB on a fresh clone. OCR quality is
    measured by scripts/evaluate.py against a labelled set, not here.
    """
    monkeypatch.setitem(sys.modules, "easyocr", None)


def reading(text, confidence=0.9, frame_id=0):
    return PlateResult(plate_text=text, confidence=confidence,
                       bbox=(0, 0, 100, 30), frame_id=frame_id)


def test_no_readings_means_no_consensus():
    engine = ANPREngine()
    assert engine.get_consensus(track_id=1) is None


def test_a_single_reading_is_its_own_consensus():
    engine = ANPREngine()
    engine.add_reading(1, reading("MH12AB1234"))
    consensus = engine.get_consensus(1)
    assert consensus.plate_text == "MH12AB1234"
    assert consensus.num_frames == 1
    assert consensus.confidence == 1.0


def test_the_majority_reading_wins():
    """The point of consensus: outvote noisy per-frame OCR errors."""
    engine = ANPREngine()
    for text in ["MH12AB1234", "MH12AB1234", "MH12A81234", "MH12AB1234", "NH12AB1234"]:
        engine.add_reading(1, reading(text))

    consensus = engine.get_consensus(1)
    assert consensus.plate_text == "MH12AB1234"
    assert consensus.num_frames == 5
    assert consensus.confidence == 3 / 5


def test_consensus_exposes_the_full_vote_tally():
    """The API returns votes so a demo can show *why* a plate won."""
    engine = ANPREngine()
    for text in ["MH12AB1234", "MH12AB1234", "MH12A81234"]:
        engine.add_reading(1, reading(text))

    votes = engine.get_consensus(1).votes
    assert votes == {"MH12AB1234": 2, "MH12A81234": 1}
    assert sum(votes.values()) == 3


def test_unanimous_readings_score_full_confidence():
    engine = ANPREngine()
    for _ in range(4):
        engine.add_reading(1, reading("MH12AB1234"))
    assert engine.get_consensus(1).confidence == 1.0


def test_tracks_vote_independently():
    engine = ANPREngine()
    engine.add_reading(1, reading("MH12AB1234"))
    engine.add_reading(2, reading("DL01CA9999"))
    assert engine.get_consensus(1).plate_text == "MH12AB1234"
    assert engine.get_consensus(2).plate_text == "DL01CA9999"


def test_consensus_serializes_with_the_fields_the_api_returns():
    engine = ANPREngine()
    engine.add_reading(1, reading("MH12AB1234"))
    as_dict = engine.get_consensus(1).to_dict()
    for key in ("plate_text", "confidence", "num_frames", "votes"):
        assert key in as_dict


def test_indian_plate_pattern_accepts_valid_plates():
    for plate in ["MH12AB1234", "DL01C1234", "KA05MN9876"]:
        assert ANPREngine.INDIAN_PATTERN.match(plate), plate


def test_indian_plate_pattern_rejects_obvious_non_plates():
    for text in ["HELLO", "123456", "A1"]:
        assert not ANPREngine.INDIAN_PATTERN.match(text), text


def test_cleanup_drops_stale_readings():
    engine = ANPREngine()
    engine.add_reading(1, reading("MH12AB1234", frame_id=5))
    engine.add_reading(2, reading("DL01CA9999", frame_id=500))
    engine.cleanup_old(max_age=100)
    assert 1 not in engine.readings
    assert 2 in engine.readings


# --- localizer selection ------------------------------------------------

def test_default_localizer_is_the_contour_fallback():
    """No model configured must still work — that is the Tier-2 CPU path."""
    engine = ANPREngine()
    assert engine.plate_model_path is None
    assert engine.localizer == "contour"


def test_contour_localizer_runs_without_any_model(frame):
    engine = ANPREngine()
    boxes = engine.detect_plate_region(frame)
    assert isinstance(boxes, list)   # a flat frame yields no candidates


def test_a_broken_model_path_degrades_to_contours(no_easyocr, capsys):
    """A missing checkpoint must not take the pipeline down with it."""
    engine = ANPREngine(plate_model_path="/nonexistent/plate.pt").load()
    assert engine.localizer == "contour"
    assert "contour fallback" in capsys.readouterr().out


def test_the_trained_localizer_is_used_when_loaded(monkeypatch, frame):
    """With a model loaded, localization must come from it, not contours."""
    class FakeBox:
        xyxy = [type("T", (), {"tolist": lambda self: [10.0, 20.0, 110.0, 50.0]})()]

    class FakeResult:
        boxes = [FakeBox()]

    engine = ANPREngine()
    engine.plate_model = lambda *args, **kwargs: [FakeResult()]

    assert engine.localizer == "yolo"
    assert engine.detect_plate_region(frame) == [(10, 20, 100, 30)]


def test_localizer_inference_failure_falls_back_per_frame(frame, capsys):
    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    engine = ANPREngine()
    engine.plate_model = explode
    boxes = engine.detect_plate_region(frame)

    assert isinstance(boxes, list)   # contour path ran instead of raising
    assert "falling back to contours" in capsys.readouterr().out


def test_read_plate_returns_nothing_without_an_ocr_reader(frame):
    engine = ANPREngine()   # load() not called, so reader is None
    assert engine.read_plate(frame, (0, 0, 100, 30)) is None
