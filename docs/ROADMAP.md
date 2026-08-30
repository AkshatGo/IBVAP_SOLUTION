# IBVAP — Roadmap

> What's not built yet, what needs fine-tuning, and the dataset/training plan.
> For what actually works, see `README.md` and `ARCHITECTURE.md`.

---

## 1. Not Yet Built (by priority)

### P0 — High Impact, Moderate Effort
| Feature | Description | Estimated Effort |
|---|---|---|
| **MQTT broker integration** | Edge nodes publish events to MQTT broker; backend subscribes | 2-3 days |
| **PostgreSQL event persistence** | Replace in-memory hash chain with database-backed storage | 2-3 days |
| **Fine-tuned detection model** | IDD + ExDark fine-tuning for Indian road scenarios | 1 week |
| **ONNX model export** | Export fine-tuned models to ONNX for edge deployment | 1 day |

### P1 — Medium Impact
| Feature | Description | Estimated Effort |
|---|---|---|
| **JWT auth + RBAC** | Operator/Commander/Admin/Auditor roles | 3-4 days |
| **TLS encryption** | HTTPS for API, encrypted storage | 1-2 days |
| **Multi-camera fan-out** | Process 4+ cameras per edge node concurrently | 1 week |
| **Store-and-forward queue** | Offline buffering, sync when link restores | 3-5 days |

### P2 — Future / Grand Finale
| Feature | Description | Estimated Effort |
|---|---|---|
| **TensorRT/Jetson optimization** | Benchmark and optimize for Jetson Orin Nano | 1-2 weeks |
| **React+Leaflet dashboard** | Multi-site map, real-time alerts | 1 week |
| **Cross-camera correlation** | Track objects across multiple cameras | 2 weeks |
| **Behavioral baseline model** | Learn normal patterns, flag anomalies | 2-3 weeks |
| **FRS-at-range** | Face detection at distance (not recognition) | 2 weeks |

---

## 2. Dataset Catalog

### 2.1 Priority Datasets (for fine-tuning)

| Priority | Dataset | Purpose | Size | Why |
|---|---|---|---|---|
| **1** | **IDD (Indian Driving Dataset)** — Detection subset | Person/vehicle detection on Indian roads | ~2.5 GB | Directly matches deployment domain — Indian roads, mixed traffic, non-Western vehicle types (autos, tempos) that COCO-pretrained YOLO under-detects |
| **2** | **Indian Number Plates** + **UFPR-ALPR** | Plate localizer fine-tuning | ~4.5 GB | Current ANPR uses classical CV for plate localization — weakest accuracy component; trained YOLO plate-detector fixes it directly |
| **3** | **ExDark** | Low-light detection robustness | ~0.8 GB | Answers the "honest night claim" — model trained on dark imagery raises real detection confidence at night |

### 2.2 All Cataloged Datasets

| Dataset | Short | Category | Size | Status |
|---|---|---|---|---|
| Indian Driving Dataset | IDD | Vehicle Detection | 2.5 GB | Priority 1 |
| IDD Temporal | IDD-T | Temporal Tracking | 5.0 GB | Cut for now |
| IITM-HeTra | IITM | Surveillance | 1.2 GB | Cut for now |
| Indian Number Plates | INP | ANPR | 3.0 GB | Priority 2 |
| UFPR-ALPR | UFPR | ANPR | 1.5 GB | Priority 2 |
| VIRAT | VIRAT | Surveillance | 50.0 GB | Cut (too large) |
| MEVA | MEVA | Surveillance | 470.0 GB | Cut (too large) |
| UCF-Crime | UCF | Anomaly | 15.0 GB | Cut for now |
| ExDark | ExDark | Low Light | 0.8 GB | Priority 3 |
| BDD100K | BDD | Vehicle Tracking | 100.0 GB | Cut (too large) |
| WIDER FACE | WIDER | Face Detection | 3.5 GB | Optional |
| Custom Border | — | Custom | TBD | Not collected yet |

### 2.3 Explicitly Cut (state as future work)

- **VIRAT** (50 GB) — not realistically downloadable or trainable in timeframe
- **MEVA** (470 GB) — same
- **BDD100K** (100 GB) — same
- **UCF-Crime** — anomaly detection is a different pipeline, not our focus
- **IITM-HeTra** — overhead CCTV, different camera angle than border deployment
- **IDD-Temporal** — temporal consistency is P2 work
- **Custom Border** — doesn't exist yet; collecting it is a multi-week program

---

## 3. Preprocessing Pipeline

All priority datasets converge on **one unified YOLO format** so a single fine-tuning run can consume them.

### 3.1 Target Directory Structure

```
data/
  detection/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt      # YOLO format: class cx cy w h (normalized)
    data.yaml                     # class names + paths
  anpr/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt      # YOLO format, single class: "plate"
    data.yaml
```

### 3.2 IDD Preprocessing

1. Download IDD-Detection subset (bounding-box annotated)
2. Convert IDD's XML annotations → YOLO `.txt` labels
3. Map classes to: person, bicycle, car, motorcycle, bus, truck
4. Normalize bounding boxes to [0,1]
5. Train/val split: 85/15
6. Deduplicate near-identical dashcam frames (perceptual hash filter)
7. **Script:** `scripts/idd_to_yolo.py`

### 3.3 ANPR Preprocessing

1. Convert both datasets to YOLO format (single class: `plate`)
2. Apply augmentation:
   - Motion blur (kernel 3-9px)
   - Perspective warp (±15°)
   - Random brightness/contrast (±30%)
   - Downscale-then-upscale (0.5x-0.7x)
   - Gaussian/Poisson sensor noise
3. Train/val split: 80/20
4. Keep classical contour localizer as CPU fallback
5. **Script:** `scripts/anpr_augmentation.py`

### 3.4 ExDark Preprocessing

1. Map ExDark classes to target classes
2. Merge with darkened subset of IDD (gamma 0.3-0.6, noise, contrast reduction)
3. Train/val split: 80/20 (genuinely dark-only held-out val)
4. One model for full lighting spectrum (not separate day/night models)

---

## 4. Training Plan

| Model | Base | Epochs | Image Size | Batch | Notes |
|---|---|---|---|---|---|
| Detection (IDD + darkened-IDD + ExDark) | `yolov8n.pt` | 50-80, early stopping | 640 | 16 | Freeze backbone first 10 epochs, then unfreeze |
| Plate localizer | `yolov8n.pt`, single-class | 50 | 640 | 16 | Watch for overfitting on smaller dataset |

**Compute:** Free-tier Kaggle (30 GPU-hrs/week) or Google Colab T4 is sufficient.

### Evaluation Metrics (for Grand Finale)

- **Detection:** mAP@0.5 and mAP@0.5:0.95, broken out for day vs night subsets
- **ANPR:** Plate localization mAP@0.5, end-to-end OCR exact match rate (before vs after fine-tuning)

---

## 5. Integration Plan

1. Export fine-tuned models to ONNX (`yolo export format=onnx`)
2. Swap `EdgeDetector` model path via config flag (A/B stock vs fine-tuned)
3. Re-run test suite against new models (fine-tuned models may shift confidence distributions)
4. Re-validate detection threshold (0.45) against new model

---

## 6. Future Dependencies (not in requirements.txt yet)

| Package | Purpose | When |
|---|---|---|
| albumentations | ANPR augmentation pipeline | When running `scripts/anpr_augmentation.py` |
| SQLAlchemy + psycopg2 | PostgreSQL persistence | P0 backend work |
| redis-py | Alert queue caching | P0 backend work |
| paho-mqtt | MQTT broker integration | P0 backend work |
| PyJWT + OAuth2 libs | Authentication | P1 auth work |
| onnxruntime | ONNX model inference | P1 edge optimization |
| TensorRT | Jetson optimization | P2 hardware tuning |
| imagehash | IDD frame deduplication | When running `scripts/idd_to_yolo.py` |
