# IBVAP — Intelligent Border Video Analytics Platform
## Product Requirements Document (PRD)
### Smart India Hackathon 2026

---

## 1. Executive Summary

**IBVAP** is a tiered, bandwidth-honest, alert-that-can-be-trusted surveillance software layer for existing CCTV infrastructure at Indian border posts and checkpoints.

> "Every AI-CCTV platform on the market assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three. IBVAP is the first platform designed for the actual constraint — not the demo constraint."

---

## 2. Problem Statement

### Background
- Border surveillance relies on legacy CCTV systems with varying connectivity
- Remote Border Out Posts (BOPs) have thin, intermittent, sometimes satellite-only links
- Existing AI-CCTV solutions assume reliable internet, modern hardware, and centralized processing
- False-positive fatigue causes guards to ignore or disable alerts
- No existing solution addresses the bandwidth, hardware, and trust constraints simultaneously

### Core Challenges
| Challenge | Impact | Current Gap |
|-----------|--------|-------------|
| Low/intermittent bandwidth | Video streaming to cloud fails | All existing solutions stream video |
| Legacy hardware | No AI-capable cameras | Solutions assume modern IP cameras |
| False-positive fatigue | Guards ignore alerts | No behavioral baseline or context |
| Night/poor visibility | Face recognition fails | Overclaiming IR/thermal capabilities |
| Security evasion | Jammed cameras go silent | No signal-loss detection |
| Data governance | FRS/ANPR near borders is sensitive | No access control or audit trail |

---

## 3. Target Users

| User | Role | Primary Need |
|------|------|--------------|
| Border Guard / Operator | Real-time monitoring | Simple, trustworthy alerts with context |
| Command Center Officer | Multi-site oversight | Aggregated view, cross-camera correlation |
| System Administrator | Deployment & maintenance | Tiered deployment, remote management |
| Security Auditor | Compliance & investigation | Tamper-evident logs, access control |

---

## 4. Functional Requirements

### 4.1 Detection & Tracking (P0 — Must Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001 | Human detection and tracking across frames | P0 |
| FR-002 | Vehicle detection and tracking across frames | P0 |
| FR-003 | Multi-object tracking with unique IDs (ByteTrack) | P0 |
| FR-004 | Virtual fence intrusion detection (user-drawn polygon) | P0 |
| FR-005 | ANPR with multi-frame OCR consensus voting | P0 |

### 4.2 Alert System (P0 — Must Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-010 | Real-time alert generation with severity levels | P0 |
| FR-011 | Explainable alerts (reason for firing) | P0 |
| FR-012 | Signal-loss-is-itself-an-alert design | P0 |
| FR-013 | Alert dashboard with map view | P0 |
| FR-014 | Alert history and filtering | P0 |

### 4.3 Edge Processing (P0 — Must Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-020 | Edge inference at camera site (Tier-1) | P0 |
| FR-021 | Store-and-forward for intermittent connectivity | P0 |
| FR-022 | Only event metadata travels upstream (not video) | P0 |
| FR-023 | Local event queuing when link is down | P0 |

### 4.4 Security & Governance (P0 — Must Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-030 | Tamper-evident hash-chained event log | P0 |
| FR-031 | Role-based access control | P0 |
| FR-032 | Data retention policy enforcement | P0 |
| FR-033 | Audit trail for all access | P0 |

### 4.5 Advanced Features (P1 — Should Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-040 | Behavioral baseline learning | P1 |
| FR-041 | Cross-camera correlation (heuristic v1) | P1 |
| FR-042 | Face detection (not full FRS at range) | P1 |
| FR-043 | C2 system integration via MQTT | P1 |

### 4.6 Roadmap Features (P2 — Could Have)
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-050 | Full FRS against watchlist (requires IR hardware) | P2 |
| FR-051 | Recognition-grade re-identification | P2 |
| FR-052 | Advanced behavioral anomaly detection | P2 |

---

## 5. Non-Functional Requirements

### 5.1 Performance
| Metric | Target | Notes |
|--------|--------|-------|
| Fence intrusion alert latency | < 3 seconds | Time-critical |
| Vehicle/ANPR match latency | < 15 seconds | Consensus voting needs window |
| Behavioral deviation alert | < 30 seconds | Requires trend evaluation |
| Signal loss alert | < 5 seconds | Near-instant check |
| Edge inference FPS | ≥ 15 FPS | YOLOv8-nano on Jetson |
| Concurrent camera streams | ≥ 4 per edge node | Tier-1 deployment |

### 5.2 Reliability
| Metric | Target | Notes |
|--------|--------|-------|
| Uptime | 99.9% | Edge nodes operate independently |
| Data durability | Zero loss | Store-and-forward, local first |
| Graceful degradation | Network down → local alerts | Never goes silent |

### 5.3 Scalability
| Metric | Target | Notes |
|--------|--------|-------|
| Border posts supported | 100+ | Via tiered architecture |
| Cameras per post | 1-20 | Varies by tier |
| Event storage | 90 days retention | Configurable |
| Concurrent dashboard users | 50+ | WebSocket-based |

### 5.4 Security
| Requirement | Implementation |
|-------------|----------------|
| Transport encryption | TLS 1.3 for all upstream comms |
| Event integrity | SHA-256 hash chain |
| Access control | RBAC with JWT tokens |
| Data at rest | AES-256 for sensitive event data |
| Audit logging | Immutable append-only log |

---

## 6. Architecture Overview

### Tiered Deployment Model

```
TIER 1 — High-priority BOP / Check Post
├── IP Camera → Local Edge Box (Jetson Orin Nano / mini-PC, ~$150-250)
│   ├── Detection (YOLOv8-nano)
│   ├── Tracking (ByteTrack)
│   ├── ANPR (PaddleOCR/EasyOCR)
│   ├── Virtual Fence Monitor
│   └── Behavioral Baseline Model
├── Outputs: JSON event + 3-5 sec clip snippet (only on trigger)
└── MQTT publish to Command Dashboard

TIER 2 — Remote Border-Road Camera
├── IP Camera → Lightweight motion trigger ($20-30 microcontroller)
│   └── Frame-differencing only
├── On trigger: buffer short clip locally
├── Forward when link exists
└── Analytics run centrally / nearest Tier-1 node

TIER 3 — Command Center (Regional/Central)
├── Aggregates all events across posts
├── Map-based dashboard
├── Cross-camera correlation
├── Long-term storage
├── Audit log
└── C2 system integration (MQTT/API)
```

### Key Design Principles
1. **Edge-first processing** — Inference happens at the camera site, not the cloud
2. **Metadata-only transport** — Only compact JSON events travel upstream, never raw video
3. **Store-and-forward** — Every tier logs locally first, syncs opportunistically
4. **Signal-loss-as-alert** — A blinded/jammed camera triggers escalation, not silence
5. **Honest capability claims** — State limitations proactively (night FRS, re-ID)

---

## 7. Alert Object Schema

```json
{
  "event_id": "e7f1...",
  "prev_hash": "a92c...",
  "timestamp": "2026-08-24T21:14:03Z",
  "site_id": "BOP-14",
  "camera_id": "CAM-2",
  "event_type": "fence_intrusion",
  "object_class": "person",
  "track_id": "T-0042",
  "zone": "Zone-3",
  "bearing": "NE",
  "speed_mps": 1.4,
  "confidence": 0.91,
  "baseline_deviation": true,
  "clip_ref": "s3://.../e7f1.mp4",
  "severity": "high",
  "explanation": "Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE. No scheduled patrol active in this zone at this time."
}
```

---

## 8. Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Detection/Tracking | YOLOv8-nano (ONNX/TensorRT) + ByteTrack | Fast, accurate, edge-deployable |
| ANPR | YOLO plate localization + PaddleOCR | Multi-frame consensus voting |
| Edge Runtime | Jetson Orin Nano / x86 mini-PC | Tier-dependent |
| Transport | MQTT | Lightweight, low-bandwidth, IoT standard |
| Backend | FastAPI + PostgreSQL + Redis | Fast, async, event-driven |
| Dashboard | React + Leaflet/Mapbox | Interactive map view |
| Tamper-evidence | SHA-256 hash chain | Simple, effective, no blockchain overhead |

---

## 9. Demo Script (5-minute version)

1. **One-liner pitch** (10 sec) — The constraint-aware platform
2. **Tiered architecture diagram** (20 sec) — Why it's deployable, not just demoable
3. **Live: Virtual fence** — Draw polygon, walk across, alert fires in < 3 sec with explanation
4. **Live: Multi-frame ANPR** — Show consensus voting beating single-frame baseline
5. **Kill a camera feed** — Signal-loss alert fires immediately as high severity
6. **Tamper-evident log** — Break a past record, show chain validation fail
7. **Honest roadmap slide** — What needs IR hardware, what we didn't fake

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Detection accuracy (mAP@0.5) | ≥ 85% | On test dataset |
| False positive rate | < 5% | Alerts per 1000 frames |
| ANPR accuracy (multi-frame) | ≥ 90% | On test plates |
| Alert latency (P95) | < 5 sec | End-to-end |
| Dashboard load time | < 2 sec | Initial load |
| Edge inference time | < 67ms/frame | 15 FPS target |

---

## 11. Constraints & Assumptions

### Constraints
- Must work with existing CCTV infrastructure (no camera replacement)
- Must operate on limited bandwidth (satellite links)
- Must handle power intermittency at remote sites
- Must comply with data governance regulations near national borders

### Assumptions
- IP cameras provide RTSP/ONNX-compatible streams
- Minimum 1 camera per border post
- At least intermittent connectivity (not permanently offline)
- Edge hardware has sufficient power (solar + battery for Tier-2)

---

## 12. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Core Detection | Week 1-2 | YOLOv8 detection, ByteTrack, virtual fence |
| Phase 2: ANPR & Alerts | Week 2-3 | Multi-frame ANPR, alert system, dashboard |
| Phase 3: Edge Deployment | Week 3-4 | Edge runtime, MQTT, store-and-forward |
| Phase 4: Security & Governance | Week 4-5 | Hash chain, RBAC, audit trail |
| Phase 5: Integration & Demo | Week 5-6 | C2 integration, demo prep, documentation |

---

## 13. Open Questions

1. What is the exact specification of existing CCTV cameras at border posts?
2. Is there an existing C2 system API specification for integration?
3. What are the specific data retention requirements per regulations?
4. Are there existing datasets for border surveillance (anonymized)?
5. What is the expected power budget for Tier-2 remote sites?

---

*Document Version: 1.0*
*Last Updated: 2026-08-29*
*Author: SIH2026 Team*
