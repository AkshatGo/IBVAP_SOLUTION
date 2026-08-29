"""
IBVAP Web Demo
Deployable web application for SIH 2026 presentation
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List
import tempfile
import os

# Page config
st.set_page_config(
    page_title="IBVAP - Intelligent Border Video Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #58a6ff;
    }
    .alert-high {
        background: #3d1f1f;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f44336;
        margin: 5px 0;
    }
    .alert-critical {
        background: #2d1a3d;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #9c27b0;
        margin: 5px 0;
    }
    .alert-medium {
        background: #3d2f1f;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 5px 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================

if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'prev_hash' not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'objects' not in st.session_state:
    st.session_state.objects = [
        {"id": 1, "class": "person", "x": 100, "y": 400, "vx": 2, "vy": 0.5, "in_fence": False},
        {"id": 2, "class": "vehicle", "x": 900, "y": 500, "vx": -3, "vy": 0, "plate": "BR12AB3456"},
        {"id": 3, "class": "person", "x": 700, "y": 350, "vx": 1, "vy": 1, "in_fence": False},
    ]
if 'camera_status' not in st.session_state:
    st.session_state.camera_status = {"CAM-01": True, "CAM-02": True, "CAM-03": True}


# ============================================================
# Helper Functions
# ============================================================

def create_alert(event_type: str, explanation: str, severity: str, obj_class: str = "person", track_id: int = 0) -> Dict:
    """Create a new alert with hash chain"""
    alert = {
        "event_id": f"e{len(st.session_state.alerts) + 1:04d}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "object_class": obj_class,
        "track_id": f"T-{track_id:04d}",
        "severity": severity,
        "explanation": explanation
    }
    
    # Update hash chain
    alert_copy = alert.copy()
    alert_copy.pop("prev_hash", None)
    st.session_state.prev_hash = hashlib.sha256(
        json.dumps(alert_copy, sort_keys=True).encode()
    ).hexdigest()
    
    st.session_state.alerts.append(alert)
    return alert


def verify_hash_chain() -> bool:
    """Verify the entire hash chain"""
    alerts = st.session_state.alerts
    for i in range(1, len(alerts)):
        prev_event = alerts[i - 1].copy()
        prev_event.pop("prev_hash", None)
        calculated_hash = hashlib.sha256(
            json.dumps(prev_event, sort_keys=True).encode()
        ).hexdigest()
        if calculated_hash != alerts[i]["prev_hash"]:
            return False
    return True


def generate_frame(width: int = 800, height: int = 500) -> np.ndarray:
    """Generate a simulated border surveillance frame"""
    st.session_state.frame_count += 1
    
    # Create base frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky
    for y in range(250):
        blue = int(80 + (y / 250) * 50)
        frame[y, :] = [blue, blue - 10, blue - 25]
    
    # Ground
    frame[250:, :] = [35, 55, 35]
    
    # Border fence
    cv2.line(frame, (30, 250), (width - 30, 250), (100, 100, 100), 3)
    
    # Virtual fence
    fence_points = np.array([[250, 200], [650, 200], [650, 450], [250, 450]], np.int32)
    cv2.polylines(frame, [fence_points], True, (0, 255, 0), 3)
    cv2.putText(frame, "VIRTUAL FENCE", (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Update objects
    for obj in st.session_state.objects:
        obj["x"] += obj["vx"]
        obj["y"] += obj["vy"]
        
        # Bounce
        if obj["x"] < 50 or obj["x"] > width - 50:
            obj["vx"] = -obj["vx"]
        if obj["y"] < 100 or obj["y"] > height - 50:
            obj["vy"] = -obj["vy"]
        
        # Check fence
        point = np.array([obj["x"], obj["y"]])
        obj["in_fence"] = cv2.pointPolygonTest(fence_points, (float(obj["x"]), float(obj["y"])), False) >= 0
        
        x, y = int(obj["x"]), int(obj["y"])
        
        if obj["class"] == "person":
            color = (0, 0, 255) if obj["in_fence"] else (0, 200, 255)
            cv2.circle(frame, (x, y - 35), 10, color, -1)
            cv2.line(frame, (x, y - 25), (x, y + 15), color, 2)
            cv2.line(frame, (x, y + 15), (x - 10, y + 30), color, 2)
            cv2.line(frame, (x, y + 15), (x + 10, y + 30), color, 2)
            cv2.rectangle(frame, (x - 25, y - 50), (x + 25, y + 40), color, 2)
            status = "ALERT" if obj["in_fence"] else "OK"
            cv2.putText(frame, f"Person {status}", (x - 25, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        elif obj["class"] == "vehicle":
            cv2.rectangle(frame, (x - 50, y - 20), (x + 50, y + 20), (100, 100, 150), -1)
            cv2.rectangle(frame, (x - 35, y - 32), (x + 35, y - 20), (80, 80, 120), -1)
            cv2.rectangle(frame, (x - 22, y + 5), (x + 22, y + 18), (255, 255, 255), -1)
            cv2.putText(frame, obj["plate"], (x - 20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.rectangle(frame, (x - 60, y - 40), (x + 60, y + 28), (255, 165, 0), 2)
            cv2.putText(frame, f"Vehicle {obj['plate']}", (x - 60, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    
    # HUD
    cv2.rectangle(frame, (0, 0), (width, 40), (15, 15, 25), -1)
    cv2.putText(frame, "IBVAP", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (88, 166, 255), 2)
    cv2.putText(frame, "LIVE", (width - 60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("IBVAP")
    st.caption("Intelligent Border Video Analytics Platform")
    
    st.divider()
    
    st.subheader("SIH 2026")
    st.markdown("**Problem Statement:** SIH26187")
    st.markdown("AI-Based Intelligent Video Analytics for Border Surveillance")
    
    st.divider()
    
    st.subheader("Demo Controls")
    
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert(
            "fence_intrusion",
            "Person T-0001 crossed virtual fence Zone-1 at 1.4 m/s, bearing NE",
            "high",
            "person",
            1
        )
        st.success("Alert generated!")
    
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        create_alert(
            "anpr_match",
            "Vehicle BR12AB3456 detected at Checkpoint-1",
            "medium",
            "vehicle",
            2
        )
        st.success("ANPR alert generated!")
    
    if st.button("📡 Simulate Signal Loss", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert(
            "signal_loss",
            "Camera CAM-03 signal lost for 3+ seconds",
            "critical",
            "none",
            0
        )
        st.success("Signal loss alert generated!")
    
    if st.button("🔄 Restore Camera", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
        st.success("CAM-03 restored!")
    
    st.divider()
    
    # Hash chain verification
    chain_valid = verify_hash_chain()
    if chain_valid:
        st.success("🔒 Hash Chain: VALID")
    else:
        st.error("⚠️ Hash Chain: TAMPERED")
    
    st.metric("Total Alerts", len(st.session_state.alerts))


# ============================================================
# Main Content
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🛡️ IBVAP - Intelligent Border Video Analytics Platform</h1>
    <p style="color: #8b949e; margin: 5px 0 0 0;">Real-time border surveillance with AI-powered detection and alerts</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Cameras", "3/3")
with col2:
    st.metric("Detection FPS", "30")
with col3:
    high_alerts = sum(1 for a in st.session_state.alerts if a["severity"] in ["high", "critical"])
    st.metric("High/Critical Alerts", high_alerts)
with col4:
    st.metric("Frame Count", st.session_state.frame_count)

# Main content
col_video, col_alerts = st.columns([3, 2])

with col_video:
    st.subheader("📹 Live Camera Feed")
    
    # Generate and display frame
    frame = generate_frame(800, 500)
    st.image(frame, channels="BGR", use_container_width=True)
    
    # Camera status
    st.subheader("📷 Camera Status")
    cam_cols = st.columns(3)
    for i, (cam, status) in enumerate(st.session_state.camera_status.items()):
        with cam_cols[i]:
            if status:
                st.success(f"✅ {cam}: Online")
            else:
                st.error(f"❌ {cam}: Offline")

with col_alerts:
    st.subheader("🚨 Alert Log")
    
    if not st.session_state.alerts:
        st.info("No alerts yet. Use the sidebar to simulate events.")
    else:
        # Show recent alerts (newest first)
        for alert in reversed(st.session_state.alerts[-10:]):
            severity = alert["severity"]
            css_class = f"alert-{severity}"
            
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{alert['event_type'].upper()}</strong><br>
                <small>{alert['explanation']}</small><br>
                <small style="color: #888;">{alert['timestamp'][:19]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Hash chain verification
    st.subheader("🔐 Security")
    chain_valid = verify_hash_chain()
    if chain_valid:
        st.success("✅ Hash chain verified - all events are tamper-proof")
    else:
        st.error("❌ Hash chain integrity compromised!")
    
    st.caption(f"Chain hash: {st.session_state.prev_hash[:32]}...")


# ============================================================
# Technical Details (Expander)
# ============================================================

with st.expander("📋 Technical Details", expanded=False):
    st.subheader("Architecture")
    st.markdown("""
    | Component | Technology |
    |-----------|------------|
    | Detection | YOLOv8-nano |
    | Tracking | ByteTrack |
    | ANPR | PaddleOCR + Multi-frame Consensus |
    | Backend | FastAPI + PostgreSQL |
    | Transport | MQTT (for C2 integration) |
    | Dashboard | Streamlit / React |
    | Security | SHA-256 Hash Chain |
    """)
    
    st.subheader("Dataset Coverage")
    st.markdown("""
    | Dataset | Purpose | Scale |
    |---------|---------|-------|
    | IDD | Indian vehicle detection | 10K images, 34 classes |
    | Indian Number Plates | ANPR training | 15K HD images |
    | VIRAT | Surveillance activities | 8.5+ hours |
    | ExDark | Low-light detection | 7.3K images |
    | WIDER FACE | Face detection | 32K images |
    """)


# ============================================================
# Footer
# ============================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #8b949e;">
    <p>IBVAP - Smart India Hackathon 2026 | Problem Statement SIH26187</p>
    <p>Built for the actual constraint, not the demo constraint.</p>
</div>
""", unsafe_allow_html=True)
