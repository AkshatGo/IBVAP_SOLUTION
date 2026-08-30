"""Backend API — every REST endpoint and the live WebSocket.

create_app() takes an optional pipeline, which is what makes this testable
without models: the tests inject a pipeline whose detector is stubbed, so
no YOLO weights are downloaded and no EasyOCR reader is built.

What these tests pin is the API *contract* — status codes, response shapes,
and that an endpoint actually reaches the module underneath it. Detection
quality is a metric (scripts/evaluate.py), not an assertion.
"""
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_detection
from src.backend.api import create_app
from src.edge import signal as signal_module
from src.edge.hashchain import HashChain
from src.edge.pipeline import IBVAPPipeline

INSIDE = (320, 240)     # inside the default 640x480 fence zone

pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A model-free pipeline with an isolated chain and threshold file."""
    # Point threshold persistence at a temp file so the POST endpoint can be
    # exercised without rewriting the repo's config/signal_thresholds.json.
    monkeypatch.setattr(signal_module, "_THRESHOLDS_PATH",
                        str(tmp_path / "signal_thresholds.json"))

    pipe = IBVAPPipeline(camera_id="CAM-TEST", site_id="BOP-TEST")
    pipe.hash_chain = HashChain(path=str(tmp_path / "hashchain.jsonl"))
    pipe.setup_fence()

    pipe._staged_detections = []
    pipe.detector.detect = lambda frame: list(pipe._staged_detections)
    pipe.anpr.process_frame = lambda frame, frame_id: []
    pipe._model_loaded = True
    return pipe


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline=pipeline))


def detection_at(center, class_id=0, class_name="person", size=40):
    cx, cy = center
    half = size // 2
    return make_detection((cx - half, cy - half, cx + half, cy + half),
                          class_id=class_id, class_name=class_name, confidence=0.9)


def raise_an_alert(pipeline, count=1):
    """Drive the pipeline until `count` fence intrusions are on the chain."""
    positions = [(200, 200), (260, 200), (320, 200), (380, 200), (440, 200)]
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    for i in range(count):
        pipeline.fence._last_alert.clear()
        pipeline._staged_detections = [detection_at(positions[i % len(positions)])]
        pipeline.process_frame(frame)


# --- basics -------------------------------------------------------------

def test_root_identifies_the_service(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "IBVAP API"


def test_status_reports_pipeline_summary(client):
    body = client.get("/api/status").json()
    assert body["site_id"] == "BOP-TEST"
    assert body["camera_id"] == "CAM-TEST"
    assert body["chain_valid"] is True


def test_openapi_schema_is_served(client):
    """Swagger docs are part of the demo; a broken schema breaks them."""
    assert client.get("/openapi.json").status_code == 200


# --- alerts and chain ---------------------------------------------------

def test_alerts_are_empty_on_a_fresh_pipeline(client):
    body = client.get("/api/alerts").json()
    assert body["alerts"] == []
    assert body["total"] == 0


def test_alerts_appear_after_an_intrusion(client, pipeline):
    raise_an_alert(pipeline)
    body = client.get("/api/alerts").json()
    assert body["total"] == 1
    assert body["alerts"][0]["event_type"] == "fence_intrusion"


def test_alerts_respect_the_last_n_parameter(client, pipeline):
    raise_an_alert(pipeline, count=5)
    body = client.get("/api/alerts", params={"last_n": 2}).json()
    assert len(body["alerts"]) == 2
    assert body["total"] == 5


def test_verify_reports_a_valid_chain(client, pipeline):
    raise_an_alert(pipeline, count=3)
    body = client.get("/api/alerts/verify").json()
    assert body["is_valid"] is True
    assert body["broken_at"] is None
    assert body["total_events"] == 3


def test_verify_reports_a_tampered_chain(client, pipeline):
    """The endpoint behind the demo's tamper moment must actually detect it."""
    raise_an_alert(pipeline, count=3)
    pipeline.hash_chain.chain[1].payload["zone_name"] = "Forged Zone"

    body = client.get("/api/alerts/verify").json()
    assert body["is_valid"] is False
    assert body["broken_at"] == 1


def test_chain_returns_every_record(client, pipeline):
    raise_an_alert(pipeline, count=2)
    records = client.get("/api/chain").json()["records"]
    assert len(records) == 2
    for record in records:
        assert {"event_id", "event_type", "hash", "prev_hash"} <= set(record)


def test_chain_records_are_linked(client, pipeline):
    raise_an_alert(pipeline, count=3)
    records = client.get("/api/chain").json()["records"]
    for earlier, later in zip(records, records[1:]):
        assert later["prev_hash"] == earlier["hash"]


def test_chain_export_is_downloadable_json(client, pipeline):
    raise_an_alert(pipeline)
    response = client.get("/api/chain/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert len(response.json()) == 1


# --- cameras and signal thresholds --------------------------------------

def test_cameras_reports_registered_cameras(client, pipeline):
    pipeline.signal_detector.register_camera("CAM-TEST")
    body = client.get("/api/cameras").json()
    assert "CAM-TEST" in body
    assert "is_online" in body["CAM-TEST"]


def test_thresholds_are_readable(client):
    body = client.get("/api/signal/thresholds").json()
    for key in ("timeout_seconds", "black_threshold", "white_threshold",
                "brightness_drop_threshold", "consecutive_black_needed"):
        assert key in body


def test_thresholds_can_be_updated_live(client):
    """Live demo tuning: a POST must change behaviour without a restart."""
    response = client.post("/api/signal/thresholds", json={
        "black_threshold": 42.0,
        "consecutive_black_needed": 1,
    })
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["thresholds"]["black_threshold"] == 42.0

    # And the change is visible on the next read, not just in the response.
    assert client.get("/api/signal/thresholds").json()["black_threshold"] == 42.0


def test_updated_thresholds_reach_the_detector(client, pipeline):
    """Regression: the write and the reload must use one canonical path."""
    client.post("/api/signal/thresholds", json={
        "black_threshold": 90.0,
        "consecutive_black_needed": 1,
    })
    # A frame that is "normal" under the defaults now trips the alarm.
    message = pipeline.signal_detector.update("CAM-TEST",
                                              np.full((64, 64, 3), 50, dtype=np.uint8))
    assert message is not None and "black" in message.lower()


# --- fences -------------------------------------------------------------

def test_fences_lists_the_default_zone(client):
    zones = client.get("/api/fences").json()["zones"]
    assert len(zones) == 1


def test_a_fence_zone_can_be_added(client):
    response = client.post("/api/fences", json={
        "name": "Checkpoint",
        "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
        "severity": "critical",
        "description": "added over the API",
    })
    assert response.status_code == 200
    names = [z["name"] for z in response.json()["zones"]]
    assert "Checkpoint" in names


def test_an_added_zone_actually_detects_intrusions(client, pipeline):
    """Adding a zone over the API must affect the running pipeline."""
    client.post("/api/fences", json={
        "name": "Checkpoint",
        "polygon": [[500, 380], [630, 380], [630, 470], [500, 470]],
        "severity": "critical",
    })
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    pipeline._staged_detections = [detection_at((565, 425))]
    result = pipeline.process_frame(frame)

    assert any(i["zone_name"] == "Checkpoint" for i in result.intrusions)


def test_a_fence_zone_can_be_removed(client):
    client.post("/api/fences", json={
        "name": "Temporary",
        "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
    })
    assert client.delete("/api/fences/Temporary").status_code == 200
    names = [z["name"] for z in client.get("/api/fences").json()["zones"]]
    assert "Temporary" not in names


def test_adding_a_zone_without_a_name_is_rejected(client):
    """A malformed body must not 500."""
    response = client.post("/api/fences", json={"polygon": [[0, 0], [1, 1], [2, 2]]})
    assert response.status_code >= 400


# --- tracks and plates --------------------------------------------------

def test_tracks_are_empty_before_any_detection(client):
    assert client.get("/api/tracks").json()["tracks"] == []


def test_tracks_report_trajectory_speed_and_bearing(client, pipeline):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    for step in range(4):
        pipeline._staged_detections = [detection_at((200 + step * 4, 200))]
        pipeline.process_frame(frame)

    tracks = client.get("/api/tracks").json()["tracks"]
    assert len(tracks) == 1
    assert {"track_id", "trajectory", "speed", "bearing"} <= set(tracks[0])
    assert tracks[0]["bearing"] == "E"


def test_plates_are_empty_without_readings(client):
    assert client.get("/api/plates").json()["plates"] == []


def test_plates_expose_the_per_frame_readings_behind_the_consensus(client, pipeline):
    """The Q&A feature: show *why* the consensus won, not just the result."""
    from src.edge.anpr import PlateResult

    for i, text in enumerate(["MH12AB1234", "MH12AB1234", "MH12A81234"]):
        pipeline.anpr.add_reading(7, PlateResult(plate_text=text, confidence=0.8,
                                                 bbox=(0, 0, 100, 30), frame_id=i))

    plates = client.get("/api/plates").json()["plates"]
    assert len(plates) == 1
    entry = plates[0]
    assert entry["plate_text"] == "MH12AB1234"
    assert entry["votes"] == {"MH12AB1234": 2, "MH12A81234": 1}
    assert len(entry["readings"]) == 3
    assert [r["frame_id"] for r in entry["readings"]] == [0, 1, 2]


# --- video upload -------------------------------------------------------

def test_uploading_a_video_returns_a_frame_count(client, tmp_path):
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (640, 480))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available in this OpenCV build")
    for _ in range(5):
        writer.write(np.full((480, 640, 3), 128, dtype=np.uint8))
    writer.release()

    with open(video_path, "rb") as handle:
        response = client.post("/api/process/video",
                               files={"file": ("clip.mp4", handle, "video/mp4")},
                               params={"max_frames": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["total_frames"] > 0
    assert "summary" in body and "alerts" in body


def test_uploading_no_file_is_rejected(client):
    assert client.post("/api/process/video").status_code == 422


# --- websocket ----------------------------------------------------------

def encode_frame(frame) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode()


def test_websocket_responds_to_ping(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "pong"}


def test_websocket_returns_an_annotated_frame_and_results(client):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "frame", "data": encode_frame(frame)}))
        message = ws.receive_json()

    assert message["type"] == "result"
    assert message["frame_id"] == 1
    for key in ("frame", "detections", "alerts", "intrusions", "plates", "chain_head"):
        assert key in message

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(message["frame"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded.shape == frame.shape


def test_websocket_streams_alerts_as_they_happen(client, pipeline):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    pipeline._staged_detections = [detection_at(INSIDE)]

    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "frame", "data": encode_frame(frame)}))
        message = ws.receive_json()

    assert len(message["intrusions"]) == 1
    assert any(a["event_type"] == "fence_intrusion" for a in message["alerts"])
    assert message["chain_head"] != ""


def test_websocket_ignores_an_undecodable_frame(client):
    """Garbage in must not kill the socket."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "frame",
                                 "data": base64.b64encode(b"not an image").decode()}))
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "pong"}
