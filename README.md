# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

> **Smart India Hackathon 2026 | Problem Statement SIH26187**
> *"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three. IBVAP is designed for the actual constraint."*

---

## 🎯 What It Does

IBVAP is a **tiered, bandwidth-honest, alert-that-can-be-trusted** surveillance software layer for existing CCTV cameras at border checkpoints and border roads.

### Core Features

| Feature | Description |
|---------|-------------|
| **Object Detection** | YOLOv8-nano for real-time person & vehicle detection |
| **Object Tracking** | ByteTrack for persistent IDs across frames |
| **Virtual Fence** | User-drawn polygon zones with intrusion detection |
| **ANPR** | Multi-frame OCR consensus voting for Indian plates |
| **Signal Loss Alert** | Camera jamming/tampering detection |
| **Tamper-Evident Log** | SHA-256 hash chain for all events |
| **Night Detection** | Motion-silhouette detection (no IR/thermal claimed) |
| **Alert Dashboard** | Real-time map, alerts, camera status |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TIER 1: Edge Node                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ YOLOv8   │→ │ ByteTrack│→ │ Virtual Fence    │  │
│  │ Detector │  │ Tracker  │  │ + ANPR + Signal  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                      ↓                              │
│              Hash Chain Logger                      │
└──────────────────────┬──────────────────────────────┘
                       │ MQTT / WebSocket / REST
┌──────────────────────┴──────────────────────────────┐
│                   TIER 2: Backend                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ FastAPI  │  │ WebSocket│  │ Event Storage    │  │
│  │ REST API │  │ Stream   │  │ + Audit Log      │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│                   TIER 3: Dashboard                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Live Feed│  │ Alert    │  │ Camera Status    │  │
│  │ + Detect │  │ Log      │  │ + Hash Verify    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
python main.py dashboard
# Opens at http://localhost:8501
```

### 3. Run API Server

```bash
python main.py server
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Run Demo on Video

```bash
python main.py demo --video path/to/video.mp4
```

### 5. Docker

```bash
docker-compose up dashboard   # Dashboard only
docker-compose up              # Full stack
```

---

## 📁 Project Structure

```
IBVAP_SOLUTION/
├── main.py                    # Entry point
├── web_demo.py                # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Full stack
│
├── src/
│   ├── config.py              # System configuration
│   │
│   ├── edge/                  # Edge inference modules
│   │   ├── detector.py        # YOLOv8 object detection
│   │   ├── tracker.py         # ByteTrack object tracking
│   │   ├── anpr.py            # ANPR with OCR consensus
│   │   ├── fence.py           # Virtual fence zones
│   │   ├── signal.py          # Signal loss detection
│   │   ├── hashchain.py       # SHA-256 tamper-evident log
│   │   └── pipeline.py        # Main inference pipeline
│   │
│   ├── backend/               # API server
│   │   └── api.py             # FastAPI REST + WebSocket
│   │
│   ├── models/                # ML model configs
│   │   └── dataset_catalog.py # Dataset registry
│   │
│   └── utils/                 # Utilities
│
├── config/                    # Configuration files
├── scripts/                   # Deployment scripts
├── tests/                     # Unit tests
├── data/                      # Datasets
│   ├── indian_number_plates/  # Indian plate dataset
│   └── models/                # Trained model weights
│
└── docs/                      # Documentation
    ├── PRD.md                 # Product Requirements
    ├── ARCHITECTURE.md        # System Architecture
    ├── API_SCHEMAS.md         # API Schemas
    └── SLIDE_DECK.md          # Presentation slides
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| GET | `/api/alerts` | Recent alerts |
| GET | `/api/alerts/verify` | Verify hash chain |
| GET | `/api/cameras` | Camera status |
| GET | `/api/fences` | Virtual fence zones |
| POST | `/api/fences` | Add fence zone |
| GET | `/api/tracks` | Active tracked objects |
| GET | `/api/plates` | ANPR results |
| POST | `/api/process/video` | Process uploaded video |
| WS | `/ws/live` | Live detection stream |

---

## 📊 Datasets Used

| Dataset | Purpose |
|---------|---------|
| IDD (IIIT-H) | Indian driving scenes |
| Indian Number Plates | ANPR training |
| VIRAT | Surveillance benchmark |
| ExDark | Low-light detection |
| WIDER FACE | Face detection |
| BDD100K | Vehicle tracking |

---

## 🎤 Pitch Points

1. **"The actual constraint, not the demo constraint"** — designed for thin/intermittent/satellite links
2. **Store-and-forward everywhere** — never goes silent, degrades gracefully
3. **Signal-loss-is-itself-an-alert** — blinded camera = high priority event
4. **Tiered by budget** — Tier 2 sites cost <$50 in hardware
5. **Honest night claims** — motion/silhouette, not face-level ID (sensor physics)
6. **Tamper-evident** — SHA-256 hash chain, provable audit trail

---

## 📝 License

Smart India Hackathon 2026 — Team IBVAP
