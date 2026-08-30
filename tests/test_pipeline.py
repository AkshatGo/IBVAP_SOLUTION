"""Pipeline contract — Detection -> Tracker -> Fence -> ANPR -> HashChain.

ROADMAP §5.3 asks for exactly this: a check that the module contract still
holds after a model swap. Fine-tuned weights shift confidence
distributions, which can change what reaches the tracker and therefore what
reaches the fence, so these tests pin the *wiring* — that a detection which
crosses a fence produces an intrusion, an alert, and a chain record, in
that order — independently of which checkpoint is loaded.

The detector is stubbed. Real detection quality is a metric
(scripts/evaluate.py), not an assertion.
"""
import pytest

from conftest import make_detection
from src.edge.pipeline import IBVAPPipeline

pytestmark = pytest.mark.integration

INSIDE = (320, 240)     # inside the default 640x480 fence zone
OUTSIDE = (20, 20)      # outside it


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A pipeline with a stubbed detector, ANPR disabled, isolated chain."""
    from src.edge.hashchain import HashChain

    pipe = IBVAPPipeline(camera_id="CAM-TEST", site_id="BOP-TEST")
    pipe.hash_chain = HashChain(path=str(tmp_path / "hashchain.jsonl"))
    pipe.setup_fence()

    # No YOLO, no EasyOCR: the contract is what's under test, not the models.
    pipe.detector.detect = lambda frame: list(pipe._staged_detections)
    pipe.anpr.process_frame = lambda frame, frame_id: []
    pipe._staged_detections = []
    pipe._model_loaded = True
    return pipe


def detection_at(center, class_id=0, class_name="person", size=40):
    cx, cy = center
    half = size // 2
    return make_detection(
        (cx - half, cy - half, cx + half, cy + half),
        class_id=class_id, class_name=class_name, confidence=0.9,
    )


def test_a_frame_with_no_detections_produces_no_alerts(pipeline, frame):
    result = pipeline.process_frame(frame)
    assert result.detections == []
    assert result.intrusions == []
    assert result.alerts == []


def test_frame_result_carries_the_full_contract(pipeline, frame):
    result = pipeline.process_frame(frame)
    for field in ("frame_id", "timestamp", "detections", "tracks", "intrusions",
                  "plates", "alerts", "camera_status", "hash_chain_head"):
        assert hasattr(result, field), field


def test_frame_ids_increment(pipeline, frame):
    assert pipeline.process_frame(frame).frame_id == 1
    assert pipeline.process_frame(frame).frame_id == 2


def test_a_detection_reaches_the_tracker_with_an_id(pipeline, frame):
    pipeline._staged_detections = [detection_at(OUTSIDE)]
    result = pipeline.process_frame(frame)
    assert len(result.tracks) == 1
    assert result.tracks[0]["track_id"] > 0


def test_crossing_the_fence_produces_an_intrusion_and_an_alert(pipeline, frame):
    """The end-to-end path the whole product claims."""
    pipeline._staged_detections = [detection_at(OUTSIDE)]
    assert pipeline.process_frame(frame).intrusions == []

    pipeline._staged_detections = [detection_at(INSIDE)]
    result = pipeline.process_frame(frame)

    assert len(result.intrusions) == 1
    assert result.intrusions[0]["object_class"] == "person"
    assert any(a["event_type"] == "fence_intrusion" for a in result.alerts)


def test_an_intrusion_is_written_to_the_hash_chain(pipeline, frame):
    pipeline._staged_detections = [detection_at(INSIDE)]
    pipeline.process_frame(frame)

    assert len(pipeline.hash_chain) == 1
    assert pipeline.hash_chain.verify() == (True, None)
    assert pipeline.hash_chain[0].event_type == "fence_intrusion"


def test_the_chain_head_advances_with_each_alert(pipeline, frame):
    """A second object entering must extend the chain, not re-alert the first.

    Note the object that is already inside is deliberately still present in
    frame two: it must NOT alert again (the fence fires on the crossing),
    while the newly-arrived vehicle must.
    """
    pipeline._staged_detections = [detection_at(INSIDE)]
    first = pipeline.process_frame(frame).hash_chain_head

    pipeline.fence._last_alert.clear()          # bypass the per-zone cooldown
    pipeline._staged_detections = [
        detection_at(INSIDE),                                    # already inside
        detection_at((260, 300), class_id=2, class_name="car"),  # newly arrived
    ]
    result = pipeline.process_frame(frame)
    second = result.hash_chain_head

    assert len(result.intrusions) == 1
    assert result.intrusions[0]["object_class"] == "car"
    assert first != second
    assert pipeline.hash_chain.verify() == (True, None)


def test_the_chain_stays_valid_across_many_frames(pipeline, frame):
    """Many alerts across many frames must leave the chain verifiable."""
    positions = [(200, 200), (260, 200), (320, 200), (380, 200), (440, 200)]
    for step in range(10):
        pipeline.fence._last_alert.clear()
        # Positions cycle: the first pass creates five separate tracks that
        # each cross in; the second pass re-matches those still-live tracks,
        # which correctly produces no further events.
        pipeline._staged_detections = [detection_at(positions[step % len(positions)])]
        pipeline.process_frame(frame)

    assert len(pipeline.hash_chain) == len(positions)
    assert pipeline.hash_chain.verify() == (True, None)


def test_a_black_frame_raises_a_signal_loss_alert(pipeline, black_frame):
    """A blinded camera is itself an alert, and must be logged as critical."""
    alerts = []
    for _ in range(4):   # past consecutive_black_needed
        alerts += pipeline.process_frame(black_frame).alerts

    signal_alerts = [a for a in alerts if a["event_type"] == "signal_loss"]
    assert signal_alerts, "a fully black feed produced no signal-loss alert"
    assert signal_alerts[0]["severity"] == "critical"


def test_alert_callbacks_fire(pipeline, frame):
    seen = []
    pipeline.on_alert(seen.append)

    pipeline._staged_detections = [detection_at(INSIDE)]
    pipeline.process_frame(frame)

    assert len(seen) == 1
    assert seen[0].event_type == "fence_intrusion"


def test_a_raising_callback_does_not_break_the_pipeline(pipeline, frame):
    def explode(record):
        raise RuntimeError("downstream consumer is down")

    pipeline.on_alert(explode)
    pipeline._staged_detections = [detection_at(INSIDE)]
    result = pipeline.process_frame(frame)   # must not raise

    assert len(result.alerts) == 1


def test_summary_reports_consistent_state(pipeline, frame):
    pipeline._staged_detections = [detection_at(INSIDE)]
    pipeline.process_frame(frame)

    summary = pipeline.get_summary()
    assert summary["site_id"] == "BOP-TEST"
    assert summary["total_frames"] == 1
    assert summary["total_alerts"] == len(pipeline.hash_chain)
    assert summary["chain_valid"] is True
    assert summary["fence_zones"] == 1


def test_the_annotated_frame_keeps_the_input_shape(pipeline, frame):
    pipeline._staged_detections = [detection_at(INSIDE)]
    result = pipeline.process_frame(frame)
    assert result.annotated_frame.shape == frame.shape
    assert result.annotated_frame is not frame   # must not mutate the input


def test_the_pipeline_honours_the_configured_model_path(monkeypatch):
    """ROADMAP §5.2 — swapping weights must need no code change."""
    monkeypatch.setenv("IBVAP_DETECTION_MODEL", "runs/detect/ibvap_detection/weights/best.pt")

    import importlib
    import src.config
    import src.edge.pipeline as pipeline_module
    importlib.reload(src.config)
    importlib.reload(pipeline_module)

    try:
        assert pipeline_module.IBVAPPipeline().detector.model_path.endswith("best.pt")
    finally:
        monkeypatch.delenv("IBVAP_DETECTION_MODEL", raising=False)
        importlib.reload(src.config)
        importlib.reload(pipeline_module)


def test_the_multi_zone_preset_reports_differing_severity(pipeline, frame):
    """Different zones must produce different alert severities."""
    pipeline.fence.zones.clear()
    pipeline.fence._inside.clear()
    pipeline.fence.add_demo_preset(640, 480)
    pipeline.fence.cooldown_seconds = 0.0

    severities = set()
    for center in ((320, 180), (320, 460)):   # pedestrian zone, vehicle lane
        pipeline._staged_detections = [detection_at(center, class_id=2, class_name="car")]
        for alert in pipeline.process_frame(frame).alerts:
            if alert["event_type"] == "fence_intrusion":
                severities.add(alert["severity"])

    assert len(severities) > 1, f"all zones alerted at one severity: {severities}"
