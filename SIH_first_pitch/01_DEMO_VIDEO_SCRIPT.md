# Demo Video — Shot List

**Target: 3 minutes.** Every claim on screen is something the code actually does.

Order is by impact per second, not by architecture. Lead with the hash chain: it
is the one moment most teams cannot show at all.

---

## Before you record

```bash
cd IBVAP_SOLUTION
make install                    # core + ML deps
rm -f data/hashchain.jsonl      # start from an empty chain
```

Open three terminals and pre-run the commands once so nothing downloads on camera:

| Terminal | Purpose |
|---|---|
| A | `make verify-chain` / `make corrupt-chain` |
| B | `make run-dashboard` (Streamlit, port 8501) |
| C | `make run-server` (FastAPI, port 8000) |

Screen-record at 1080p. Terminal font size 16+ — judges watch this on a laptop.

---

## Shot 1 — Tamper-evident chain (0:00–0:40) ⭐ LEAD WITH THIS

The strongest 30 seconds you have. Nothing else in the demo is as hard to fake.

**Do:**
1. Run the pipeline briefly so events accumulate, then: `make verify-chain`
   → `✅ Chain VALID — N events verified`
2. `make corrupt-chain` → `🔓 Corrupted record at index N`
3. `make verify-chain` again
   → `❌ Chain BROKEN at record index N` **plus the offending record printed**

**Say:**
> "Every alert is hash-chained to the one before it. If anyone edits a record —
> including us — verification fails and names the exact record. This is what
> makes the log admissible rather than just stored."

**Why it lands:** it is falsifiable on camera. You break it live and the system
catches you.

---

## Shot 2 — Auto-rickshaw detection (0:40–1:20) ⭐ SECOND STRONGEST

**Do:** show `assets/autorickshaw_vs_couch.png` full screen for 5 seconds, then
run the live detector on an Indian street clip.

**Say:**
> "COCO — what every off-the-shelf model is trained on — has no auto-rickshaw
> class. It cannot detect one. On this frame stock YOLOv8 calls one a truck and
> the other a couch. We fine-tuned on the Indian Driving Dataset: both are
> auto-rickshaws, at 0.94 confidence. Auto-rickshaw is our best class,
> 0.617 mAP at IoU 0.5."

**Why it lands:** it is specific, visual, and Indian-context. The "couch" label
is genuinely memorable and no one will forget it.

---

## Shot 3 — Camera tampering (1:20–1:50)

**Do:** with the dashboard live on webcam, cover the lens with your hand. Hold 3
seconds. A **critical** signal-loss alert fires and lands in the chain.

**Say:**
> "A blinded camera is itself an alert. Three consecutive dark frames and the
> system raises critical severity and writes it to the audit chain — so an
> operator cannot quietly disable a feed."

Optional 10-second extension: `POST /api/signal/thresholds` to change
sensitivity, show it take effect with no restart.

---

## Shot 4 — Multi-zone virtual fence (1:50–2:20)

**Do:** walk into the pedestrian zone (high severity), then the vehicle lane
(medium). Two different severities from one camera.

**Say:**
> "Zones carry their own severity. A person in the restricted zone is a
> high-severity event; a vehicle in the transit lane is medium. Every crossing
> is logged with speed, bearing and an explanation string."

---

## Shot 5 — ANPR consensus (2:20–2:45)

**Do:** show `GET /api/plates` in the browser, pointing at the `votes` object.

**Say:**
> "Per-frame OCR is noisy, so we take a majority vote across frames. The API
> returns the individual readings and the tally, not just the winner — an
> operator can see why a plate was accepted. Replacing our contour-based plate
> finder with a trained localiser took F1 from 0.19 to 0.92 on held-out images."

---

## Shot 6 — Graceful degradation (2:45–3:00) — close on this

**Do:** unset the trained model and restart; it falls back to the classical
contour localiser and keeps running.

**Say:**
> "Tier-1 sites get the trained models. Tier-2 sites with no GPU fall back to
> classical CV automatically — same pipeline, no code change. The models are
> 6MB and run at 22 FPS on CPU alone."

**Why it lands:** this is your feasibility argument, demonstrated rather than
asserted.

---

## Do not show or say

- Anything on a Jetson — **no device has ever run this code**
- "TensorRT" or "edge-optimised" — ONNX is exported and verified, nothing more
- Face recognition — not built
- End-to-end ANPR accuracy — OCR is stock EasyOCR and has never been scored
- MQTT, PostgreSQL, JWT, TLS — all roadmap

---

## If you only have 60 seconds

Shot 1 (hash chain) + Shot 2 (auto-rickshaw). Those two alone carry the
submission.
