# Q&A Preparation

The principle throughout: **answer the question directly, then pivot to what you
did do.** Every hard question below has a real answer. Confident honesty reads as
competence; evasion reads as a weakness you are hiding.

Never invent a number. If you do not know, say "I don't have that measured" —
which is itself a strong answer when everything else you say is measured.

---

## The two you will definitely get

### "Have you run this on actual edge hardware?"

> **"No. No Jetson has executed this code, and I won't claim otherwise.**
> What we have measured: the model is YOLOv8-nano, 3 million parameters, 5.9MB
> on disk, and it runs at 45ms per frame — 22 FPS — on a laptop CPU with no GPU
> at all. We've exported to ONNX and verified it loads and runs through
> onnxruntime. TensorRT conversion and Jetson benchmarking are the next step,
> and they're in our roadmap with that status."

**Do not** say "edge-optimised" or quote a device FPS. The CPU number is
genuinely good — 22 FPS with no accelerator makes the Tier-2 feasibility case on
its own.

### "How big was your dataset?"

> **"Modest, and deliberately so for round one.** Detection: 8,716 images from
> IDD with 91,700 annotated boxes, held-out validation on 960 of them. Plates:
> 460 images across two sources, 91 held out. We're not claiming
> state-of-the-art — we're claiming a measured improvement over the baseline we
> replaced. The full IDD set is 33,000 images and it's downloaded; we trained on
> a subset to get a result in hours rather than overnight."

**If pressed on the small plate set:** the DataCluster "Indian Number Plates"
dataset advertised as 10,000 images is a commercial product — Kaggle hosts a
47-image free sample. Say that. It shows you verified your sources rather than
trusting a description.

---

## On the results

### "0.508 mAP isn't very high, is it?"

> "For 7 classes on 8,000 images with a nano model, it's a reasonable first
> result — and the number I'd point at isn't the average. Auto-rickshaw is our
> best class at 0.617, which matters because COCO has no such class at all.
> Our weakest is bicycle at 0.309, and the reason is visible in the data: 614
> training instances against 35,000 persons. More data fixes that, and we know
> exactly how much we need."

Naming your own weakest class before they find it is disarming and reads as
rigour.

### "Why is recall only 0.44?"

> "That's at the default threshold. We swept it — F1 peaks at confidence 0.25,
> not the 0.45 our config originally shipped, and we changed the default because
> for a border deployment a missed intruder costs more than a false alert. The
> sweep is a script in the repo: `scripts/evaluate.py threshold`."

### "How do I know you didn't test on your training data?"

> "Strict held-out splits — 80/20 for plates, IDD's own val split for detection.
> The plate baseline comparison runs both localisers over the same 91 images
> neither was trained on. The split code and the evaluation script are both in
> the repo."

---

## On the technology

### "Why not use a bigger model?"

> "The deployment target decides that. A border post runs on constrained
> hardware, so nano-sized weights aren't a compromise — they're the requirement.
> 5.9MB and 22 FPS on CPU is what makes Tier-2 sites viable. If Tier-1 hardware
> budget allows, moving to yolov8s is a config change, not a rewrite."

### "What happens if the model fails to load, or there's no GPU?"

> "It degrades rather than dies — that's built in and tested. Detection falls
> back from YOLO to classical CV to motion detection. ANPR falls back from the
> trained localiser to contour detection, which needs no model and no GPU. Same
> pipeline, no code change. I can demo it by unsetting an environment variable."

### "Isn't the hash chain just a log file?"

> "Each record contains the hash of the previous one, so editing any record
> invalidates every record after it. The verifier reports the exact index where
> the chain breaks. I can demonstrate it — corrupt a record live and watch it get
> caught. It's persisted to disk, so it survives a restart, which an in-memory
> log wouldn't."

### "What about face recognition?"

> **"We don't do it and we don't claim it.** At border-camera distances, face
> recognition would be unreliable, and presenting an unreliable identification as
> evidence is worse than not having it. We detect and track people; identification
> is out of scope."

---

## On scope

### "How much of this actually works versus is planned?"

> "The README has a table that separates exactly that, and the roadmap lists
> everything unbuilt with effort estimates. Working: detection, tracking, virtual
> fence, ANPR, tamper detection, hash chain, the API, the dashboard. Not built:
> MQTT, PostgreSQL persistence, JWT auth, TensorRT, cross-camera correlation."

Offering the honest table *unprompted* is one of the strongest moves available.

### "What was hardest?"

> "Finding the bugs that don't announce themselves. Our tracker was assigning a
> new ID every frame — which silently broke fence entry/exit, speed estimation,
> and ANPR consensus at once, because they all depend on a stable ID. Nothing
> crashed. We now have 159 tests, and the regression tests were verified by
> checking they fail against the old code."

### "Who did what?" / "How much is AI-generated?"

Answer honestly for your team. If AI tooling was used, saying so plainly and
then demonstrating that you understand the system is far stronger than being
caught out. The test suite and the measured results are yours either way.

---

## Numbers to have memorised

| | |
|---|---|
| Plate localiser F1 | 0.915 (from 0.194) — **4.7×** |
| Detection mAP@0.5 | 0.508 overall |
| Auto-rickshaw mAP@0.5 | 0.617 — best class |
| Model size | 5.9 MB, 3.0M params |
| CPU inference | 45 ms — 22 FPS |
| Detection training data | 8,716 images, 91.7k boxes |
| Plate training data | 460 images |
| Tests | 159 |

---

## The one-sentence summary

> "We built a working edge pipeline, fine-tuned two models on Indian data,
> measured both against the baselines they replaced, and documented exactly what
> isn't built yet."
