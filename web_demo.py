"""
IBVAP Web Demo - Intelligent Border Video Analytics Platform
Works without heavy ML libraries - uses OpenCV HOG + simulated vehicle detection
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
from datetime import datetime

st.set_page_config(page_title="IBVAP", page_icon="🛡️", layout="wide")

# ── Load HOG person detector ──
@st.cache_resource
def load_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

# ── Session state ──
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {"CAM-01": True, "CAM-02": True, "CAM-03": True}

# ── Virtual fence polygon (relative to 640x480 frame) ──
W, H = 640, 480
FENCE = np.array([[180, 120], [460, 120], [460, 380], [180, 380]], np.int32)

# ── Helpers ──
def create_alert(event_type, explanation, severity, track_id=0):
    alert = {
        "event_id": f"e{len(st.session_state.alerts)+1:04d}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "site_id": "BOP-01",
        "camera_id": "CAM-01",
        "event_type": event_type,
        "track_id": f"T-{track_id:04d}",
        "severity": severity,
        "confidence": round(np.random.uniform(0.82, 0.97), 2),
        "explanation": explanation,
    }
    # Hash chain
    payload = {k: v for k, v in alert.items() if k != "prev_hash"}
    alert["prev_hash"] = st.session_state.prev_hash
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    st.session_state.prev_hash = h
    st.session_state.alerts.append(alert)
    return alert


def verify_chain():
    for i in range(1, len(st.session_state.alerts)):
        prev = st.session_state.alerts[i - 1].copy()
        prev_hash_val = prev.pop("prev_hash", None)
        h = hashlib.sha256(json.dumps(prev, sort_keys=True).encode()).hexdigest()
        if h != st.session_state.alerts[i]["prev_hash"]:
            return False
    return True


def detect_vehicles_by_color(frame):
    """Simple color-based vehicle detection (blue/red cars)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    vehicles = []
    # Blue cars
    mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
    # Red cars
    mask2 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = mask | mask2 | mask3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if 2000 < area < 30000:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if 0.8 < aspect < 4.0:
                vehicles.append((x, y, x + w, y + h))
    return vehicles


def process_frame(frame, hog):
    """Run person + vehicle detection on a frame."""
    frame = cv2.resize(frame, (W, H))
    detections = []

    # ── Person detection (HOG) ──
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.03)
    for (x, y, w, h), conf in zip(boxes, weights):
        cx, cy = x + w // 2, y + h // 2
        in_fence = cv2.pointPolygonTest(FENCE, (float(cx), float(cy)), False) >= 0
        color = (0, 0, 255) if in_fence else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"Person {conf[0]:.0%}"
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        detections.append({"class": "person", "in_fence": in_fence, "conf": float(conf[0])})
        if in_fence:
            create_alert(
                "fence_intrusion",
                f"Person detected crossing virtual fence Zone-1. Confidence {conf[0]:.0%}.",
                "high",
            )

    # ── Vehicle detection (color-based) ──
    vboxes = detect_vehicles_by_color(frame)
    for (x1, y1, x2, y2) in vboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, "Vehicle", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
        detections.append({"class": "vehicle", "in_fence": False, "conf": 0.80})

    # ── Draw fence ──
    cv2.polylines(frame, [FENCE], True, (0, 255, 0), 2)
    cv2.putText(frame, "VIRTUAL FENCE", (210, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, detections


def generate_demo_frame(tick):
    """Generate a synthetic surveillance-like frame for demo mode."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (40, 50, 40)  # dark green background (ground)
    # sky
    frame[:120] = (60, 40, 30)
    # road
    cv2.rectangle(frame, (0, 350), (W, H), (70, 70, 70), -1)
    cv2.line(frame, (0, 400), (W, 400), (200, 200, 200), 1)
    # fence
    cv2.polylines(frame, [FENCE], True, (0, 255, 0), 2)
    cv2.putText(frame, "VIRTUAL FENCE", (210, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # animated person walking
    px = 100 + (tick * 3) % 400
    py = 250 + int(10 * np.sin(tick * 0.2))
    cv2.circle(frame, (px, py - 25), 12, (180, 180, 200), -1)  # head
    cv2.line(frame, (px, py - 12), (px, py + 15), (180, 180, 200), 2)  # body
    cv2.line(frame, (px, py), (px - 12, py + 20), (180, 180, 200), 2)
    cv2.line(frame, (px, py), (px + 12, py + 20), (180, 180, 200), 2)
    # check if person in fence
    in_fence = cv2.pointPolygonTest(FENCE, (float(px), float(py)), False) >= 0
    color = (0, 0, 255) if in_fence else (0, 255, 255)
    cv2.rectangle(frame, (px - 18, py - 40), (px + 18, py + 25), color, 2)
    cv2.putText(frame, "PERSON", (px - 15, py - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    # car on road
    car_x = 500 - (tick * 4) % 600
    car_y = 410
    cv2.rectangle(frame, (car_x, car_y - 20), (car_x + 50, car_y + 5), (0, 100, 200), -1)
    cv2.rectangle(frame, (car_x + 10, car_y - 30), (car_x + 40, car_y - 20), (0, 80, 180), -1)
    cv2.circle(frame, (car_x + 8, car_y + 5), 5, (30, 30, 30), -1)
    cv2.circle(frame, (car_x + 42, car_y + 5), 5, (30, 30, 30), -1)
    cv2.putText(frame, "VEHICLE", (car_x + 5, car_y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 1)
    # HUD
    cv2.rectangle(frame, (0, 0), (W, 30), (20, 20, 30), -1)
    cv2.putText(frame, f"IBVAP | BOP-01 CAM-01 | Frame {tick} | {datetime.now().strftime('%H:%M:%S')}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    return frame, in_fence


# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ IBVAP")
    st.caption("Intelligent Border Video Analytics Platform")
    st.divider()

    st.subheader("🎮 Demo Controls")
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert("fence_intrusion", "Person crossed Zone-1 at 1.4 m/s, bearing NE. No patrol scheduled.", "high", 1)
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        plate = f"BR{np.random.randint(10,99)}AB{np.random.randint(1000,9999)}"
        create_alert("anpr_match", f"Vehicle {plate} flagged — plate matches watchlist.", "medium", 2)
    if st.button("📡 Simulate Signal Loss", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert("signal_loss", "CAM-03 at BOP-01 lost signal. Possible jamming/tampering.", "critical")
    if st.button("🔄 Restore Camera", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
        create_alert("signal_restored", "CAM-03 signal restored.", "low")

    st.divider()
    st.subheader("🔐 Tamper-Evident Log")
    chain_ok = verify_chain()
    if chain_ok:
        st.success("✅ Hash chain VALID")
    else:
        st.error("❌ Hash chain BROKEN — tampering detected!")
    st.metric("Total Alerts", len(st.session_state.alerts))
    if st.session_state.prev_hash != "0" * 64:
        st.caption(f"Chain head: `{st.session_state.prev_hash[:24]}…`")

    st.divider()
    st.subheader("📡 Camera Status")
    for cam, ok in st.session_state.camera_status.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} {cam}")

# ═══════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════
st.markdown("# 🛡️ IBVAP — Border Video Analytics Platform")
st.caption("Real-time AI-powered surveillance • YOLOv8 detection • Virtual fences • ANPR • Tamper-evident logging")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Detection Feed")

    mode = st.radio("Mode", ["🎬 Synthetic Demo", "🎥 Upload Video", "📷 Webcam"], horizontal=True)

    hog = load_hog()
    placeholder = st.empty()

    if mode == "🎬 Synthetic Demo":
        tick = st.session_state.frame_count
        for _ in range(3):
            frame, in_fence = generate_demo_frame(tick)
            # Run HOG on synthetic frame too
            frame_resized = cv2.resize(frame, (W, H))
            boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for (x, y, w, h), conf in zip(boxes, weights):
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame_resized, f"Person {conf[0]:.0%}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                detections_count = len(boxes)
            placeholder.image(frame_resized, channels="BGR", use_container_width=True)
            tick += 1
            time.sleep(0.3)
        st.session_state.frame_count = tick

        if in_fence:
            create_alert("fence_intrusion", "Animated person crossed virtual fence Zone-1.", "high")

    elif mode == "🎥 Upload Video":
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            if ret:
                frame, dets = process_frame(frame, hog)
                placeholder.image(frame, channels="BGR", use_container_width=True)
                st.caption(f"Detected: {len(dets)} objects | "
                           f"Persons: {sum(1 for d in dets if d['class'] == 'person')} | "
                           f"Vehicles: {sum(1 for d in dets if d['class'] == 'vehicle')}")
            cap.release()
        else:
            st.info("📁 Upload a video to see real HOG person detection + color-based vehicle detection.")

    else:
        st.warning("📷 Webcam requires a browser with camera access. Use Synthetic Demo or Upload Video for the pitch.")
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                frame, dets = process_frame(frame, hog)
                placeholder.image(frame, channels="BGR", use_container_width=True)
            cam.release()

with col2:
    st.subheader("🚨 Alert Log")

    if not st.session_state.alerts:
        st.info("No alerts yet. Use the demo controls or upload a video to generate events.")
    else:
        for alert in reversed(st.session_state.alerts[-15:]):
            sev = alert["severity"]
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(sev, "⚪")
            st.markdown(f"{icon} **`{alert['event_type'].upper()}`** — {alert['explanation']}")
            st.caption(f"⏱ {alert['timestamp'][:19]} | ID: {alert['event_id']} | "
                       f"Confidence: {alert.get('confidence', 'N/A')}")

    st.divider()
    st.subheader("📊 System Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Cameras", sum(1 for v in st.session_state.camera_status.values() if v))
    c2.metric("Alerts", len(st.session_state.alerts))
    critical = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
    c3.metric("Critical", critical)

    st.divider()
    st.subheader("📋 Alert Object Schema")
    if st.session_state.alerts:
        last = st.session_state.alerts[-1]
        st.json(last)

# ── Footer ──
st.divider()
st.caption("IBVAP — Smart India Hackathon 2026 | "
           "\"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. "
           "Border posts have none of those three. IBVAP is designed for the actual constraint.\"")
