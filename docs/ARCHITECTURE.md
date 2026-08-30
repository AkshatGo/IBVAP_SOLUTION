# IBVAP — System Architecture

> Ground-truth document: describes what is **actually implemented and runnable today**.
> For what's planned but not built, see `ROADMAP.md`.

---

## 1. Entry Points

```
                              main.py
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        "server"            "dashboard"           "demo"
              │                  │                  │
              ▼                  ▼                  ▼
   src/backend/api.py      subprocess:         src/edge/pipeline.py
   create_app()        streamlit run          IBVAPPipeline
   → FastAPI on :8000    web_demo.py            .process_frame()
   → IBVAPPipeline()      → :8501               on video/webcam
```

**Default run (`python main.py`) starts the dashboard.**

---

## 2. Edge Modules (`src/edge/`)

### 2.1 EdgeDetector (`detector.py`)
- **Engine:** YOLOv8-nano via ultralytics
- **Fallback:** OpenCV HOG person detector (if ultralytics unavailable)
- **Target classes:** person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
- **Confidence threshold:** 0.45
- **Input:** 640×640 frame
- **Output:** `List[Detection]` with track_id=-1 (assigned by tracker)

### 2.2 ObjectTracker (`tracker.py`)
- **Method:** Greedy IoU association (ByteTrack-inspired, no external dependencies)
- **Key properties:**
  - Persistent track IDs (monotonically increasing, never reused)
  - Same-class-only matching
  - Track aging: unmatched tracks drop after `track_buffer` (30) frames
  - Confirmation gate: `min_hits=3` before a track is trusted
- **Outputs:** trajectory history, speed (px/s), compass bearing (N/S/E/W)
- **Dependencies:** numpy only

### 2.3 VirtualFence (`fence.py`)
- **Method:** Polygon-based intrusion detection via `cv2.pointPolygonTest`
- **Features:**
  - Multiple named zones with severity levels
  - Entry/exit tracking per object per zone
  - Cooldown (default 5s) to prevent alert spam
  - Speed and bearing in intrusion explanation
- **Dependencies:** OpenCV only

### 2.4 ANPREngine (`anpr.py`)
- **Plate detection:** Contour analysis → quadrilateral filter → aspect ratio (2:1–6:1) → area filter
- **OCR engine:** EasyOCR (English, CPU mode)
- **Consensus:** Majority voting across N frames (confidence = votes/total)
- **Indian plate pattern:** `[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}`
- **OCR error correction:** O→0, I→1, Z→2, S→5
- **Dependencies:** OpenCV, easyocr

### 2.5 SignalLossDetector (`signal.py`)
- **4 heuristics:**
  1. Timeout: no frame for N seconds
  2. Black frame: brightness < 5.0 (3 consecutive frames)
  3. White frame: brightness > 250.0 (sensor failure)
  4. Brightness drop: sudden >30% drop (jamming)
- **Key design:** A blinded camera is itself a critical alert
- **Dependencies:** numpy only

### 2.6 HashChain (`hashchain.py`)
- **Algorithm:** SHA-256, append-only JSONL file
- **Structure:** Each record's hash includes the previous record's hash
- **Verification:** `verify()` walks the full chain, returns (is_valid, broken_at_index)
- **Persistence:** Writes to `data/hashchain.jsonl`, survives restarts
- **Dependencies:** stdlib (hashlib, json, os)

### 2.7 IBVAPPipeline (`pipeline.py`)
Orchestrates all modules in sequence per frame:
```
Frame
  → SignalLossDetector.update()     → [signal_loss alert if triggered]
  → EdgeDetector.detect()           → List[Detection]
  → ObjectTracker.update()          → List[Detection] with track_ids
  → VirtualFence.check_intrusion()  → [fence_intrusion alerts if triggered]
  → ANPREngine.process_frame()      → [anpr_match alerts if consensus ≥ 0.7]
  → HashChain.add_event()           → SHA-256 linked record
  → FrameResult (annotated frame + all results)
```

---

## 3. Backend (`src/backend/api.py`)

FastAPI application wrapping a single `IBVAPPipeline` instance.

### REST Endpoints
| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | API name, version, status |
| `/api/status` | GET | Pipeline summary (frames, alerts, chain status) |
| `/api/alerts` | GET | Last N alerts from hash chain |
| `/api/alerts/verify` | GET | Chain integrity check |
| `/api/cameras` | GET | Camera online/offline status |
| `/api/fences` | GET/POST/DELETE | Fence zone management |
| `/api/tracks` | GET | Active tracked objects + trajectories |
| `/api/plates` | GET | ANPR consensus results |
| `/api/chain` | GET | Full hash chain |
| `/api/chain/export` | GET | Export chain as JSON download |
| `/api/process/video` | POST | Process uploaded video file |

### WebSocket
- `/ws/live` — Accepts base64-encoded frames, returns annotated frame + detection results + alerts

---

## 4. Dashboard (`web_demo.py`)

Self-contained Streamlit app (does NOT import `src/edge/*` — reimplements detection inline).

### Modes
1. **Demo Mode** — Synthetic surveillance frames with animated person/car
2. **Upload Video** — Frame-by-frame analysis with slider
3. **Webcam** — Live camera feed

### Features
- YOLOv8 detection with HOG fallback
- Virtual fence overlay
- ANPR plate detection
- Alert log with severity icons
- Hash chain verification
- Camera status display
- Demo simulation controls (fence intrusion, ANPR match, signal loss)

---

## 5. Configuration (`src/config.py`)

Dataclass-based configuration with 7 sections:
- `DetectionConfig` — model path, confidence, NMS, target classes
- `TrackingConfig` — thresholds, buffer, frame rate
- `ANPRConfig` — OCR engine, consensus frames, plate pattern
- `FenceConfig` — default polygon, cooldown, zone names
- `AlertConfig` — severity levels, hash algorithm, retention
- `CameraConfig` — resolution, FPS, signal loss timeout
- `ServerConfig` — host, port, CORS

Global singleton: `CONFIG = IBVAPConfig()`

---

## 6. Logging (`src/utils/logger.py`)

Structured logger with methods: `info()`, `warning()`, `error()`, `alert()`, `detection()`, `signal_loss()`, `chain()`.

---

## 7. Docker Deployment

```bash
# Dashboard only
docker-compose up dashboard

# Full stack
docker-compose up
```

- `Dockerfile` — Python 3.10-slim, OpenCV system deps, pre-downloads YOLOv8 model
- `docker-compose.yml` — Two services: `dashboard` (:8501) and `api` (:8000)
