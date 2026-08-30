# Slide Deck — Measured Results

Every number here is measured on this repository and reproducible. Nothing is
projected, rounded up, or borrowed from a paper.

---

## Slide 1 — Title

**IBVAP — Intelligent Border Video Analytics Platform**
Edge-first video analytics for border surveillance

`github.com/AkshatGo/IBVAP_SOLUTION`

---

## Slide 2 — The problem

Border camera feeds are watched by people, and people miss things. Three
specific failures:

1. **Nobody is watching at 3am.** Intrusions go unseen until review.
2. **A disabled camera looks like a quiet camera.** Tampering is invisible.
3. **Off-the-shelf detection does not understand Indian roads.** Models trained
   on Western datasets have never seen an auto-rickshaw.

---

## Slide 3 — The one-slide proof ⭐

> Full-bleed image: `assets/autorickshaw_vs_couch.png`

**Left — stock YOLOv8n (COCO):** `truck 0.80`, `couch 0.49`
**Right — our fine-tune (IDD):** `autorickshaw 0.94`, `autorickshaw 0.94`

COCO has no auto-rickshaw class, so an off-the-shelf model *cannot* label one
correctly — it forces it into the nearest wrong category.

*This is the slide people remember. Let it breathe.*

---

## Slide 4 — What we built

Detection → Tracking → Virtual fence → ANPR → Tamper-evident chain

| Component | State |
|---|---|
| Person/vehicle detection | Fine-tuned on IDD, 7 classes |
| Multi-object tracking | Dependency-free IoU tracker, stable IDs |
| Multi-zone virtual fence | Per-zone severity, entry/exit events |
| ANPR | Trained plate localiser + multi-frame OCR consensus |
| Camera tamper detection | 4 heuristics, config-tunable live |
| Audit log | SHA-256 hash chain, persisted and verifiable |
| API | 16 REST endpoints + live WebSocket |

**159 automated tests**, running in ~4 seconds.

---

## Slide 5 — Measured results

### Plate localisation — replacing classical CV with a trained model

| Localiser | Precision | Recall | F1 |
|---|---|---|---|
| Contour (before) | 0.250 | 0.158 | 0.194 |
| **Trained YOLOv8n** | **0.929** | **0.901** | **0.915** |

**4.7× F1 improvement.** Recall 15.8% → 90.1%. 91 held-out images, IoU ≥ 0.5.

### Detection — fine-tuned on IDD

Overall **mAP@0.5 = 0.508** across 7 classes, 960 held-out images.

| Class | mAP@0.5 |
|---|---|
| **autorickshaw** | **0.617** |
| motorcycle | 0.602 |
| car | 0.593 |
| bus | 0.578 |
| person | 0.479 |
| truck | 0.383 |
| bicycle | 0.309 |

The class COCO does not have is the class we detect best.

---

## Slide 6 — Built for the edge

| Property | Value |
|---|---|
| Architecture | YOLOv8-nano |
| Parameters | 3,011,043 |
| Weights | 5.9 MB |
| CPU inference | 45 ms/frame — **22 FPS, no GPU** |
| GPU inference | 1.6 ms/frame |
| ONNX export | Verified through onnxruntime |

**Graceful degradation, three tiers:** trained models → classical CV → motion
detection. A site with no GPU still runs the same pipeline.

**Honest status:** ONNX is exported and verified. TensorRT and Jetson
benchmarking are roadmap — we have not run this on a device.

---

## Slide 7 — Data

| Dataset | Used for | Size |
|---|---|---|
| IDD-Detection | Person/vehicle detection | 8,716 images / 91.7k boxes |
| Kaggle car-plate-detection | Plate localisation | 433 images |
| DataCluster Indian plates | Plate localisation | 47 images (free sample) |

Both training sets use strict held-out validation — 80/20 for plates, IDD's own
val split for detection. No test image was trained on.

**Stated plainly:** these are modest datasets. The claim is a measured
before/after improvement, not a state-of-the-art result.

---

## Slide 8 — Engineering rigour

Building this surfaced six defects that were **silent** — the code ran and
produced plausible output while doing the wrong thing:

- The tracker assigned a fresh ID every frame, breaking fence entry/exit, speed,
  and ANPR consensus simultaneously
- The brightness-drop tamper check compared a value against itself and could
  never fire
- Class indices were hardcoded to COCO, so the fine-tuned model would have
  reported every auto-rickshaw as a truck

Each is now covered by a regression test that fails against the old code.

---

## Slide 9 — Roadmap

**Next:** full IDD training run · ExDark low-light fine-tune · Jetson
benchmarking + TensorRT · end-to-end ANPR accuracy against plate-string ground
truth

**Then:** MQTT edge→server transport · PostgreSQL persistence · JWT/RBAC ·
cross-camera correlation

Everything not built is listed in `docs/ROADMAP.md` rather than implied.

---

## Slide 10 — Close

**Working code, measured results, honest gaps.**

- 4.7× improvement on our weakest component, measured against its baseline
- The class COCO cannot detect is the class we detect best
- Every alert cryptographically chained and verifiable
- 6 MB models, 22 FPS on CPU alone

Live demo · repo · this deck all reproducible from `main`.
