# 🛡️ IBVAP — Intelligent Border Video Analytics Platform

> **Smart India Hackathon 2026 | Problem Statement SIH26187**
> *"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three."*

---

## What It Does

IBVAP is a **bandwidth-honest, alert-that-can-be-trusted** surveillance software layer for existing CCTV cameras at border checkpoints and border roads. All AI inference runs at the edge (no cloud dependency). Only compact event metadata travels upstream — never video.

---

## What Actually Works

| Feature | Status | Module |
|---|---|---|
| YOLOv8-nano object detection (persons, vehicles) | ✅ Working | `src/edge/detector.py` |
| Multi-object tracking with persistent IDs | ✅ Working | `src/edge/tracker.py` |
| Virtual fence intrusion detection (multiple zones) | ✅ Working | `src/edge/fence.py` |
| ANPR with multi-frame consensus voting | ✅ Working | `src/edge/anpr.py` |
| Camera signal-loss / tamper detection | ✅ Working | `src/edge/signal.py` |
| SHA-256 tamper-evident hash chain | ✅ Working (JSONL-persisted) | `src/edge/hashchain.py` |
| Full inference pipeline | ✅ Working | `src/edge/pipeline.py` |
| Streamlit dashboard (demo/upload/webcam) | ✅ Working | `web_demo.py` |
| FastAPI REST + WebSocket API | ✅ Working | `src/backend/api.py` |
| Docker deployment | ✅ Working | `Dockerfile`, `docker-compose.yml` |

### What's Documented But Not Yet Built

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full list. Key items:
- MQTT broker integration
- PostgreSQL event persistence
- JWT auth + RBAC
- TensorRT/Jetson edge optimization
- ONNX model export
- React+Leaflet multi-site dashboard
- Fine-tuned detection models (IDD, ExDark datasets)
- Cross-camera correlation

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  TIER 1: Edge Node                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ YOLOv8   │→ │ Object   │→ │ Virtual Fence + ANPR + │ │
│  │ Detector │  │ Tracker  │  │ Signal Loss Detection  │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
│                      ↓                                    │
│          Hash Chain Logger (SHA-256, JSONL)              │
└──────────────────────┬───────────────────────────────────┘
                       │ WebSocket / REST
┌──────────────────────┴───────────────────────────────────┐
│                  TIER 2: Backend                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ FastAPI  │  │ WebSocket│  │ Event Storage           │ │
│  │ REST API │  │ Stream   │  │ + Audit Log             │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────────┐
│                  TIER 3: Dashboard                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ Live Feed│  │ Alert    │  │ Camera Status + Hash    │ │
│  │ + Detect │  │ Log      │  │ Verification            │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Data Flow (per frame)

```
frame → SignalLoss → Detection → Tracking → Fence Check → ANPR → Hash Chain → Annotated Frame + Alerts
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full module descriptions.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard (default)
```bash
python main.py dashboard
# Opens at http://localhost:8501
```

### 3. Run API Server
```bash
python main.py server
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Run Demo on Video
```bash
python main.py demo --video path/to/video.mp4
```

### 5. Docker
```bash
docker-compose up dashboard    # Dashboard only
docker-compose up              # Full stack (dashboard + API)
```

### 6. Verify Hash Chain
```bash
python -m src.edge.hashchain verify
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System status |
| GET | `/api/alerts` | Recent alerts |
| GET | `/api/alerts/verify` | Verify hash chain |
| GET | `/api/cameras` | Camera status |
| GET | `/api/fences` | Virtual fence zones |
| POST | `/api/fences` | Add fence zone |
| DELETE | `/api/fences/{name}` | Remove fence zone |
| GET | `/api/tracks` | Active tracked objects |
| GET | `/api/plates` | ANPR results |
| GET | `/api/chain` | Full hash chain |
| GET | `/api/chain/export` | Export chain as JSON |
| POST | `/api/process/video` | Process uploaded video |
| WS | `/ws/live` | Live detection stream |

---

## Project Structure

```
IBVAP_SOLUTION/
├── main.py                    # Entry point (server | dashboard | demo)
├── web_demo.py                # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── Makefile                   # Dev commands
├── Dockerfile                 # Docker build
├── docker-compose.yml         # Full stack
│
├── src/
│   ├── config.py              # System configuration (dataclasses)
│   ├── edge/                  # Edge inference modules
│   │   ├── detector.py        # YOLOv8 object detection
│   │   ├── tracker.py         # Multi-object tracking (greedy IoU)
│   │   ├── anpr.py            # ANPR with OCR consensus
│   │   ├── fence.py           # Virtual fence zones
│   │   ├── signal.py          # Signal loss detection
│   │   ├── hashchain.py       # SHA-256 tamper-evident log
│   │   └── pipeline.py        # Main inference pipeline
│   ├── backend/
│   │   └── api.py             # FastAPI REST + WebSocket
│   └── utils/
│       └── logger.py          # Structured logging
│
├── scripts/                   # Dataset conversion & augmentation
│   ├── idd_to_yolo.py         # IDD-Detection → YOLO converter
│   └── anpr_augmentation.py   # Plate augmentation pipeline
│
├── data/                      # Datasets (created by scripts/)
├── docs/                      # Technical documentation
│   ├── ARCHITECTURE.md        # System architecture
│   ├── ROADMAP.md             # What's not built yet
│   └── DEPLOY.md              # Streamlit Cloud deployment
│
└── pitch/                     # Presentation assets
    ├── SLIDE_DECK.md
    ├── SIH_PITCH_GUIDE.md
    ├── DEMO_SCRIPT.md
    ├── RECORDING_GUIDE.md
    ├── FINAL_SUMMARY.md
    ├── COMPLETE_PACKAGE.md
    └── HACKATHON_TIMELINE.md
```

---

## Tech Stack

| Category | Packages |
|---|---|
| Core | numpy, opencv-python-headless |
| ML/Detection | ultralytics, torch, torchvision |
| OCR | easyocr |
| Backend | fastapi, uvicorn, python-multipart |
| Dashboard | streamlit |
| Dataset scripts | albumentations |

---

## Datasets

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full dataset catalog and fine-tuning plan. Priority datasets for fine-tuning:

| Priority | Dataset | Purpose | Size |
|---|---|---|---|
| 1 | IDD (Indian Driving Dataset) | Person/vehicle detection on Indian roads | ~2.5 GB |
| 2 | Indian Number Plates + UFPR-ALPR | Plate localizer fine-tuning | ~4.5 GB |
| 3 | ExDark | Low-light detection robustness | ~0.8 GB |

---

## Hardware Tiers

| Tier | Hardware | Cost | Capabilities |
|---|---|---|---|
| **Tier 1** | Jetson Orin Nano | ~$150-250 | Full AI: detection, tracking, ANPR, virtual fence |
| **Tier 2** | Microcontroller (MCU) | ~$20-30 | Motion trigger, store-and-forward |
| **Tier 3** | Command Center | SaaS | Multi-site aggregation, alert management |

---

## Key Design Decisions

1. **Edge-first** — All inference runs locally; only compact metadata travels upstream (never video)
2. **Signal-loss-is-itself-an-alert** — A blinded camera triggers escalation, not silence
3. **Honest night claims** — Motion/silhouette detection, not face-level ID
4. **Tamper-evident** — SHA-256 hash chain makes the audit trail provably immutable
5. **Graceful degradation** — Falls back from YOLO→HOG when GPU unavailable
6. **Budget-tiered** — Tier 2 sites cost <$50 in hardware

---

## Pitch Materials

Presentation assets, demo scripts, and hackathon guides are in the [`pitch/`](pitch/) folder.

---

## License

Smart India Hackathon 2026 — Team IBVAP
