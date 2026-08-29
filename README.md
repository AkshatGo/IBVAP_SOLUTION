# IBVAP — Intelligent Border Video Analytics Platform
### Smart India Hackathon 2026

> **"Every AI-CCTV platform on the market assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three. IBVAP is the first platform designed for the actual constraint — not the demo constraint."**

---

## 🎯 One-Line Pitch

IBVAP is a tiered, bandwidth-honest, alert-that-can-be-trusted surveillance software layer for existing CCTV infrastructure at Indian border posts and checkpoints.

---

## 📋 Problem Statement

| Challenge | Impact | Current Gap |
|-----------|--------|-------------|
| Low/intermittent bandwidth | Video streaming to cloud fails | All existing solutions stream video |
| Legacy hardware | No AI-capable cameras | Solutions assume modern IP cameras |
| False-positive fatigue | Guards ignore alerts | No behavioral baseline or context |
| Night/poor visibility | Face recognition fails | Overclaiming IR/thermal capabilities |
| Security evasion | Jammed cameras go silent | No signal-loss detection |
| Data governance | FRS/ANPR near borders is sensitive | No access control or audit trail |

---

## 🏗️ Architecture

### Tiered Deployment Model

```
TIER 1 — High-priority BOP / Check Post
├── IP Camera → Local Edge Box (Jetson Orin Nano, ~$150-250)
│   ├── Detection (YOLOv8-nano)
│   ├── Tracking (ByteTrack)
│   ├── ANPR (PaddleOCR/EasyOCR)
│   ├── Virtual Fence Monitor
│   └── Behavioral Baseline Model
└── MQTT publish to Command Dashboard

TIER 2 — Remote Border-Road Camera
├── IP Camera → Lightweight motion trigger ($20-30 MCU)
├── On trigger: buffer short clip locally
└── Forward when link exists

TIER 3 — Command Center (Regional/Central)
├── Aggregates all events across posts
├── Map-based dashboard
├── Cross-camera correlation
└── C2 system integration (MQTT/API)
```

### Key Design Principles

1. **Edge-first processing** — Inference happens at the camera site, not the cloud
2. **Metadata-only transport** — Only compact JSON events travel upstream, never raw video
3. **Store-and-forward** — Every tier logs locally first, syncs opportunistically
4. **Signal-loss-as-alert** — A blinded/jammed camera triggers escalation, not silence
5. **Honest capability claims** — State limitations proactively

---

## 🚀 Features

### Build and Demo Live (Hackathon-Realistic)
1. ✅ **Human + Vehicle Detection/Tracking** — YOLOv8-nano, real-time
2. ✅ **Virtual Fence Intrusion Detection** — User-drawn polygon, instant alert
3. ✅ **ANPR with Multi-Frame OCR Consensus** — Beats single-frame baseline
4. ✅ **Alert Dashboard** — Map view, severity colors, explanation field
5. ✅ **Signal-Loss Alerting** — Kill a camera feed, see alert fire immediately
6. ✅ **Tamper-Evident Log** — Hash chain, break a record, show chain validation fail

### Design Fully, Demo as Mock/Roadmap
7. 📋 **Face Detection → Identity Match** — Detection only, FRS requires IR hardware
8. 📋 **Behavioral Baseline Learning** — Show mechanism, not claimed weeks of learning
9. 📋 **Cross-Camera Correlation** — Heuristic v1, not full re-identification
10. 📋 **C2 Integration** — Show JSON payload and MQTT topic structure

---

## ⚡ Performance Targets

| Alert Type | Target Latency | Why |
|------------|----------------|-----|
| Virtual fence intrusion | < 3 sec | Time-critical |
| Vehicle/ANPR match | < 15 sec | Consensus voting needs window |
| Behavioral deviation | < 30 sec | Requires trend evaluation |
| Signal loss | < 5 sec | Near-instant check |

---

## 🛡️ Security

- **Signal-loss = alert** — Built-in, not an afterthought
- **Hash chain** — SHA-256 tamper-evident event log
- **RBAC** — Role-based access control (Operator, Commander, Admin, Auditor)
- **Audit trail** — Immutable append-only log
- **Data retention** — Configurable per data type

---

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| Detection/Tracking | YOLOv8-nano (ONNX/TensorRT) + ByteTrack |
| ANPR | YOLO plate localization + PaddleOCR |
| Edge Runtime | Jetson Orin Nano / x86 mini-PC |
| Transport | MQTT (lightweight, low-bandwidth) |
| Backend | FastAPI + PostgreSQL + Redis |
| Dashboard | React + Leaflet/Mapbox |
| Tamper-evidence | SHA-256 hash chain |

---

## 📁 Project Structure

```
SIH2026/
├── docs/
│   ├── PRD.md                    # Product Requirements Document
│   ├── ARCHITECTURE.md           # System architecture details
│   ├── API_SCHEMAS.md            # API schemas and data models
│   └── PROJECT_GRAPH.md          # Interactive project visualization
├── src/
│   ├── edge/                     # Edge inference modules
│   ├── backend/                  # FastAPI backend
│   ├── dashboard/                # React dashboard
│   └── models/                   # ML model configs
├── config/                       # Configuration files
├── scripts/                      # Deployment scripts
├── tests/                        # Test suite
└── data/                         # Datasets and model weights
```

---

## 🎬 Demo Script (5 minutes)

1. **One-liner pitch** (10 sec) — The constraint-aware platform
2. **Tiered architecture diagram** (20 sec) — Why it's deployable, not just demoable
3. **Live: Virtual fence** — Draw polygon, walk across, alert fires in < 3 sec with explanation
4. **Live: Multi-frame ANPR** — Show consensus voting beating single-frame baseline
5. **Kill a camera feed** — Signal-loss alert fires immediately as high severity
6. **Tamper-evident log** — Break a past record, show chain validation fail
7. **Honest roadmap slide** — What needs IR hardware, what we didn't fake

---

## 🏆 Why This Wins

| Criteria | How IBVAP Addresses It |
|----------|------------------------|
| **Innovation** | Bandwidth/hardware-tiered design + signal-loss-as-alert + explainable alerts |
| **Technical Feasibility** | Every "built" claim is actually built; reach items clearly labeled roadmap |
| **Real-World Viability** | Cost tiers, store-and-forward, cold-start handling |
| **Security-Mindedness** | Tamper-evident logs, adversarial-evasion awareness, governance |

---

## 📚 Documentation

- [Product Requirements Document](docs/PRD.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [API Schemas & Data Models](docs/API_SCHEMAS.md)
- [Project Visualization Graph](docs/PROJECT_GRAPH.md)

---

## 🤝 Team

- **SIH2026 Team**

---

## 📄 License

MIT License

---

*Last Updated: 2026-08-29*
