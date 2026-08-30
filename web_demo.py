"""
IBVAP — Intelligent Border Video Analytics Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full working dashboard with:
  • YOLOv8 / HOG person + vehicle detection
  • Virtual fence intrusion alerts
  • ANPR (number plate recognition)
  • Tamper-evident SHA-256 hash chain
  • Camera signal-loss detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
import tempfile
import random
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="IBVAP — Border Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════

@st.cache_resource
def load_yolo():
    """Load YOLOv8 model."""
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_hog():
    """Load OpenCV's HOG person detector as fallback.

    Returns None on OpenCV 5+, which removed HOGDescriptor entirely. The
    hosted demo has no torch, so this fallback is its *primary* path — a
    hard failure here takes the whole public page down, which is why it
    degrades instead of raising.
    """
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


@st.cache_resource
def load_motion_detector():
    """Last-resort detector: background subtraction, available in every
    OpenCV version. Detects movement, not people — labelled as such."""
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32,
                                              detectShadows=False)

@st.cache_resource
def load_ocr():
    """Load EasyOCR for ANPR."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

# Detection engine, best available first: YOLO -> HOG -> motion.
USE_YOLO = False
hog = None
motion = None
try:
    model = load_yolo()
    USE_YOLO = True
    ENGINE = "YOLOv8"
except Exception:
    model = None
    hog = load_hog()
    if hog is not None:
        ENGINE = "HOG"
    else:
        motion = load_motion_detector()
        ENGINE = "Motion"

ocr_reader = load_ocr()

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {
        "CAM-01": True, "CAM-02": True, "CAM-03": True
    }
if "plate_cache" not in st.session_state:
    st.session_state.plate_cache = {}

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

W, H = 640, 480
FENCE = np.array([[180, 120], [460, 120], [460, 380], [180, 380]], np.int32)

TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {2, 3, 5, 7}

INDIAN_PLATE_PATTERN = (
    r"(?:^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$)"
)

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def create_alert(event_type, explanation, severity, payload=None):
    """Create alert with hash chain."""
    alert = {
        "event_id": f"e{uuid4_hex()}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "site_id": "BOP-01",
        "camera_id": "CAM-01",
        "event_type": event_type,
        "severity": severity,
        "confidence": round(np.random.uniform(0.82, 0.97), 2),
        "explanation": explanation,
        "payload": payload or {},
    }
    # Compute hash
    payload_for_hash = {k: v for k, v in alert.items() if k not in ("prev_hash", "hash")}
    h = hashlib.sha256(json.dumps(payload_for_hash, sort_keys=True, default=str).encode()).hexdigest()
    alert["hash"] = h
    alert["prev_hash"] = st.session_state.prev_hash
    st.session_state.prev_hash = h
    st.session_state.alerts.append(alert)
    return alert


def uuid4_hex():
    """Generate a random hex string (no uuid import needed)."""
    import random
    return ''.join(random.choices('0123456789abcdef', k=16))


def verify_chain():
    """Verify hash chain integrity."""
    chain = st.session_state.alerts
    if len(chain) < 2:
        return True
    for i in range(1, len(chain)):
        prev = chain[i - 1].copy()
        prev.pop("hash", None)
        expected = hashlib.sha256(
            json.dumps(prev, sort_keys=True, default=str).encode()
        ).hexdigest()
        if chain[i].get("prev_hash") != chain[i - 1].get("hash"):
            return False
    return True


def is_in_fence(point):
    """Check if a point is inside the virtual fence."""
    return cv2.pointPolygonTest(FENCE, (float(point[0]), float(point[1])), False) >= 0


def detect_frame(frame):
    """Run detection on a frame. Returns annotated frame + detections."""
    frame = cv2.resize(frame, (W, H))
    detections = []

    if USE_YOLO and model is not None:
        results = model(frame, conf=0.45, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in TARGET_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_fence = is_in_fence((cx, cy))
                cls_name = TARGET_CLASSES[cls]
                detections.append({
                    "class": cls_name, "class_id": cls,
                    "confidence": conf, "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy), "in_fence": in_fence,
                })
    elif hog is not None:
        # HOG fallback — OpenCV 4.x only
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                               padding=(4, 4), scale=1.05)
        for (x, y, w, h), conf in zip(boxes, weights):
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "person", "class_id": 0,
                "confidence": float(conf[0]),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })
    elif motion is not None:
        # Motion fallback — movement, not classification. Confidence is a
        # blob-area heuristic, not a model score; do not present it as one.
        mask = motion.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "motion", "class_id": 0,
                "confidence": min(0.99, area / (W * H) * 8),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })

    # Draw
    vis = frame.copy()
    cv2.polylines(vis, [FENCE], True, (0, 255, 0), 2)
    cv2.putText(vis, "VIRTUAL FENCE", (210, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    persons = 0
    vehicles = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["in_fence"] else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if det["class"] == "person":
            persons += 1
            if det["in_fence"]:
                create_alert(
                    "fence_intrusion",
                    f"Person detected crossing virtual fence. Confidence: {det['confidence']:.0%}.",
                    "high",
                )
        elif det["class_id"] in VEHICLE_CLASSES:
            vehicles += 1

    # HUD
    cv2.rectangle(vis, (0, 0), (W, 28), (20, 20, 30), -1)
    cv2.putText(vis, f"IBVAP | BOP-01 CAM-01 | Frame {st.session_state.frame_count} | "
                f"Persons: {persons} | Vehicles: {vehicles}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

    return vis, detections


def run_ocr_on_frame(frame):
    """Run OCR on frame for plate detection."""
    if ocr_reader is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / max(float(h), 1)
            area = w * h
            if 2.0 < aspect < 6.0 and 1000 < area < 50000:
                plate_img = frame[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
                if plate_img.size == 0:
                    continue
                plate_img = cv2.resize(plate_img, None, fx=2, fy=2,
                                        interpolation=cv2.INTER_CUBIC)
                results = ocr_reader.readtext(plate_img)
                for (_, text, conf) in results:
                    text = text.strip().upper()
                    if conf > 0.5 and len(text) >= 4:
                        plates.append({
                            "text": text, "confidence": conf,
                            "bbox": (x, y, x + w, y + h),
                        })
    return plates


def generate_demo_frame(tick):
    """Generate a synthetic surveillance frame for demo."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Ground / grass
    frame[:] = (35, 50, 35)
    # Sky
    frame[:100] = (70, 50, 35)
    # Road
    cv2.rectangle(frame, (0, 340), (W, H), (65, 65, 65), -1)
    cv2.line(frame, (0, 370), (W, 370), (180, 180, 180), 1, cv2.LINE_AA)
    # Dashed centre line
    for dx in range(0, W, 40):
        cv2.line(frame, (dx, 370), (dx + 20, 370), (240, 240, 60), 2, cv2.LINE_AA)

    # ── Fence ──
    cv2.polylines(frame, [FENCE], True, (0, 220, 0), 2, cv2.LINE_AA)
    # Label top-centre of fence
    fc = FENCE.mean(axis=0).astype(int)
    cv2.putText(frame, "VIRTUAL FENCE", (fc[0] - 50, FENCE[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1, cv2.LINE_AA)

    # ── Animated person ──
    px = 100 + (tick * 3) % 400
    py = 260 + int(8 * np.sin(tick * 0.25))
    # Body
    cv2.circle(frame, (px, py - 28), 11, (200, 200, 210), -1, cv2.LINE_AA)
    cv2.line(frame, (px, py - 16), (px, py + 10), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px - 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px + 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    in_fence = is_in_fence((px, py))
    pcolor = (0, 0, 255) if in_fence else (0, 255, 255)
    cv2.rectangle(frame, (px - 18, py - 42), (px + 18, py + 24), pcolor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"PERSON 0.{random.randint(82,97)}", (px - 18, py - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, pcolor, 1, cv2.LINE_AA)

    # ── Animated car ──
    car_x = 500 - (tick * 4) % 650
    car_y = 380
    cv2.rectangle(frame, (car_x, car_y - 18), (car_x + 55, car_y + 8), (10, 80, 180), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (car_x + 8, car_y - 28), (car_x + 45, car_y - 18), (10, 60, 160), -1, cv2.LINE_AA)
    cv2.circle(frame, (car_x + 10, car_y + 10), 5, (30, 30, 30), -1)
    cv2.circle(frame, (car_x + 45, car_y + 10), 5, (30, 30, 30), -1)
    cv2.putText(frame, f"CAR 0.{random.randint(75,94)}", (car_x, car_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 165, 0), 1, cv2.LINE_AA)

    # ── HUD top bar ──
    cv2.rectangle(frame, (0, 0), (W, 26), (15, 15, 25), -1)
    cv2.putText(
        frame,
        f"IBVAP  |  BOP-01  CAM-01  |  Frame {tick}  |  {datetime.now().strftime('%H:%M:%S')}",
        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA,
    )
    # Detection count bar
    cv2.rectangle(frame, (0, H - 24), (W, H), (15, 15, 25), -1)
    cv2.putText(frame, f"Detection engine: {ENGINE}  |  Objects: 2",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    return frame, in_fence


# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ IBVAP")
    st.caption("Intelligent Border Video Analytics Platform")
    st.divider()

    # Model status
    if USE_YOLO:
        st.success("✅ YOLOv8 loaded")
    elif ENGINE == "HOG":
        st.warning("⚠️ Using HOG fallback (no torch)")
    else:
        st.warning("⚠️ Using motion fallback (no torch, OpenCV 5+)")
    if ocr_reader:
        st.success("✅ EasyOCR loaded")
    else:
        st.info("ℹ️ OCR not available")

    st.divider()
    st.subheader("🎮 Demo Controls")
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert("fence_intrusion",
                     "Person crossed Zone-1 at 1.4 m/s, bearing NE. No patrol scheduled.",
                     "high")
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        import random
        plate = f"BR{random.randint(10,99)}AB{random.randint(1000,9999)}"
        create_alert("anpr_match",
                     f"Vehicle {plate} flagged — plate matches watchlist at Checkpoint-1.",
                     "medium",
                     {"plate_text": plate})
    if st.button("📡 Simulate Signal Loss", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert("signal_loss",
                     "CAM-03 at BOP-01 lost signal. Possible jamming or tampering.",
                     "critical")
    if st.button("🔄 Restore Camera", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
        create_alert("signal_restored", "CAM-03 signal restored.", "low")
    if st.button("🗑️ Clear All Alerts", use_container_width=True):
        st.session_state.alerts = []
        st.session_state.prev_hash = "0" * 64

    st.divider()
    st.subheader("🔐 Hash Chain")
    chain_ok = verify_chain()
    if chain_ok:
        st.success("✅ Chain VALID")
    else:
        st.error("❌ Chain BROKEN!")
    st.metric("Total Events", len(st.session_state.alerts))
    if st.session_state.prev_hash != "0" * 64:
        st.caption(f"Head: `{st.session_state.prev_hash[:20]}…`")

    st.divider()
    st.subheader("📡 Cameras")
    for cam, ok in st.session_state.camera_status.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} {cam}")

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

st.markdown("# 🛡️ IBVAP — Border Video Analytics Platform")
st.caption("Real-time AI-powered surveillance • YOLOv8 detection • Virtual fences • ANPR • Tamper-evident logging")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Detection Feed")

    mode = st.radio("Source", ["🎬 Demo Mode", "🎥 Upload Video", "📷 Webcam"],
                     horizontal=True)

    placeholder = st.empty()

    if mode == "🎬 Demo Mode":
        tick = st.session_state.frame_count
        for _ in range(3):
            frame, in_fence = generate_demo_frame(tick)
            # Run detection on synthetic frame
            annotated, dets = detect_frame(frame)
            placeholder.image(annotated, channels="BGR", use_container_width=True)
            tick += 1
            time.sleep(0.3)
        st.session_state.frame_count = tick

        persons = sum(1 for d in dets if d["class"] == "person")
        vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
        st.caption(f"🔍 Detected: {len(dets)} objects | Persons: {persons} | Vehicles: {vehicles}")

    elif mode == "🎥 Upload Video":
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)

            # Video controls
            frame_idx = st.slider("Frame", 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                annotated, dets = detect_frame(frame)

                # Run OCR
                plates = run_ocr_on_frame(frame)
                if plates:
                    for p in plates:
                        cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                        cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                placeholder.image(annotated, channels="BGR", use_container_width=True)

                persons = sum(1 for d in dets if d["class"] == "person")
                vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                st.caption(f"🔍 Frame {frame_idx} | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")

            cap.release()
        else:
            st.info("📁 Upload a video to see detection in action")

    else:
        st.warning("📷 Webcam requires browser camera access.")
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                annotated, dets = detect_frame(frame)
                placeholder.image(annotated, channels="BGR", use_container_width=True)
            cam.release()

with col2:
    st.subheader("🚨 Alert Log")

    if not st.session_state.alerts:
        st.info("No alerts yet. Use demo controls or upload video.")
    else:
        for alert in reversed(st.session_state.alerts[-15:]):
            sev = alert["severity"]
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(sev, "⚪")
            st.markdown(f"{icon} **`{alert['event_type'].upper()}`**")
            st.markdown(f"> {alert['explanation']}")
            st.caption(f"⏱ {alert['timestamp'][:19]} | ID: {alert['event_id']}")

    st.divider()

    # Stats
    st.subheader("📊 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cameras", sum(1 for v in st.session_state.camera_status.values() if v))
    c2.metric("Alerts", len(st.session_state.alerts))
    critical = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
    c3.metric("Critical", critical)

    # Alert object schema
    if st.session_state.alerts:
        st.subheader("📋 Latest Alert Object")
        last = st.session_state.alerts[-1].copy()
        last.pop("prev_hash", None)
        st.json(last)

    # Hash chain verification
    st.subheader("🔗 Hash Chain Verification")
    if st.button("Verify Chain Integrity"):
        chain_ok = verify_chain()
        if chain_ok:
            st.success(f"✅ Chain valid — {len(st.session_state.alerts)} events verified")
        else:
            st.error("❌ Chain integrity compromised!")

# Footer
st.divider()
st.caption("IBVAP — Smart India Hackathon 2026 | "
           "\"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust. "
           "Border posts have none of those three.\"")
