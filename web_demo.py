"""
IBVAP Web Demo - Full Working Version with Real YOLOv8
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
from datetime import datetime

st.set_page_config(page_title="IBVAP", page_icon="🛡️", layout="wide")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

# Initialize session state
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {"CAM-01": True, "CAM-02": True, "CAM-03": True}

# Fence
FENCE = np.array([[250, 200], [950, 200], [950, 550], [250, 550]], np.int32)

def create_alert(event_type, explanation, severity, track_id=0):
    alert = {
        "event_id": f"e{len(st.session_state.alerts)+1:04d}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "track_id": f"T-{track_id:04d}",
        "severity": severity,
        "explanation": explanation
    }
    h = hashlib.sha256(json.dumps({k:v for k,v in alert.items() if k!="prev_hash"}, sort_keys=True).encode()).hexdigest()
    st.session_state.prev_hash = h
    st.session_state.alerts.append(alert)
    return alert

def verify_chain():
    for i in range(1, len(st.session_state.alerts)):
        prev = st.session_state.alerts[i-1].copy()
        prev.pop("prev_hash", None)
        h = hashlib.sha256(json.dumps(prev, sort_keys=True).encode()).hexdigest()
        if h != st.session_state.alerts[i]["prev_hash"]:
            return False
    return True

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("IBVAP")
    st.caption("Intelligent Border Video Analytics")
    st.divider()
    
    st.subheader("🎮 Demo Controls")
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert("fence_intrusion", "Person crossed Zone-1 at 1.4 m/s", "high", 1)
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        create_alert("anpr_match", "Vehicle BR12AB3456 at Checkpoint-1", "medium", 2)
    if st.button("📡 Signal Loss CAM-03", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert("signal_loss", "CAM-03 signal lost", "critical")
    if st.button("🔄 Restore CAM-03", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
    
    st.divider()
    chain_ok = verify_chain()
    st.success("🔒 Hash Chain: VALID") if chain_ok else st.error("⚠️ Hash Chain: TAMPERED")
    st.metric("Total Alerts", len(st.session_state.alerts))

# Main
st.markdown("# 🛡️ IBVAP - Border Video Analytics Platform")
st.caption("Real-time AI-powered surveillance with YOLOv8 detection, virtual fences, ANPR, and tamper-evident logging")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Detection")
    
    # Upload or use webcam
    source = st.radio("Source", ["Upload Video", "Webcam"], horizontal=True)
    
    if source == "Upload Video":
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = None
    else:
        cap = cv2.VideoCapture(0)
    
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            model = load_model()
            
            # Run YOLOv8 detection
            results = model(frame, conf=0.45, verbose=False)
            
            # Process results
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    
                    # Only persons (0) and vehicles (2,3,5,7)
                    if cls in [0, 2, 3, 5, 7]:
                        cls_name = "person" if cls == 0 else "vehicle"
                        in_fence = cv2.pointPolygonTest(FENCE, (float(cx), float(cy)), False) >= 0
                        
                        color = (0, 0, 255) if in_fence else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{cls_name} {conf:.0%}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detections.append({"class": cls_name, "in_fence": in_fence, "bbox": (x1, y1, x2, y2)})
                        
                        if cls_name == "person" and in_fence:
                            create_alert("fence_intrusion", f"Person crossed Zone-1", "high")
            
            # Draw fence
            cv2.polylines(frame, [FENCE], True, (0, 255, 0), 3)
            cv2.putText(frame, "VIRTUAL FENCE", (350, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            st.image(frame, channels="BGR", use_container_width=True)
            st.caption(f"Detected: {len(detections)} objects | Persons: {sum(1 for d in detections if d['class']=='person')} | Vehicles: {sum(1 for d in detections if d['class']=='vehicle')}")
        else:
            st.warning("Could not read frame")
        cap.release()
    else:
        # Show demo image
        st.info("Upload a video or enable webcam to see real YOLOv8 detection in action!")
        
        # Static demo
        demo_frame = np.zeros((500, 800, 3), dtype=np.uint8)
        demo_frame[:] = [35, 55, 35]
        cv2.rectangle(demo_frame, (0, 0), (800, 40), (15, 15, 25), -1)
        cv2.putText(demo_frame, "IBVAP - Upload video to see detection", (100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2)
        cv2.polylines(demo_frame, [np.array([[200,150],[600,150],[600,400],[200,400]])], True, (0,255,0), 3)
        cv2.putText(demo_frame, "VIRTUAL FENCE", (280, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        st.image(demo_frame, channels="BGR", use_container_width=True)

with col2:
    st.subheader("🚨 Alert Log")
    for alert in reversed(st.session_state.alerts[-10:]):
        sev = alert["severity"]
        color = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(sev, "⚪")
        st.markdown(f"{color} **{alert['event_type'].upper()}** — {alert['explanation']}")
        st.caption(alert["timestamp"][:19])
    
    st.subheader("🔐 Security")
    st.success("✅ Hash chain verified") if verify_chain() else st.error("❌ Hash chain broken!")
    st.caption(f"Hash: {st.session_state.prev_hash[:32]}...")

# Footer
st.divider()
st.caption("IBVAP — SIH 2026 | Built for the actual constraint, not the demo constraint")
