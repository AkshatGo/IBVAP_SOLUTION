# SIH — First Pitch Pack

Everything needed for the round-1 submission, with every number traceable to a
command in this repository.

| File | What it is |
|---|---|
| `01_DEMO_VIDEO_SCRIPT.md` | Shot-by-shot recording script, ~3 min, ordered by impact |
| `02_SLIDE_DECK.md` | Deck content with the measured results |
| `03_QA_PREP.md` | Honest answers to the hard questions |
| `assets/autorickshaw_vs_couch.png` | **The slide.** Stock YOLOv8 vs fine-tuned |
| `assets/autorickshaw_vs_bus.png` | Second comparison, if you want a backup |

## The three numbers that carry the pitch

1. **Plate localiser F1: 0.194 → 0.915** — 4.7×, measured against the classical
   detector it replaced, on 91 held-out images.
2. **Auto-rickshaw mAP@0.5: 0.617** — our best class, and one COCO does not have.
3. **5.9 MB, 22 FPS on CPU** — the edge feasibility argument, with no GPU.

## Regenerating the figures

```bash
python scripts/make_comparison_figure.py \
    --image data/detection/images/val/<stem>.jpg \
    --out SIH_first_pitch/assets/autorickshaw_vs_couch.png
```

## Reproducing the numbers

```bash
# Plate localiser vs contour baseline
python scripts/evaluate.py localizer \
    --plate-model models/weights/plate.pt

# Detection, per class
python scripts/evaluate.py detection \
    --weights models/weights/detection.pt --data data/detection/data.yaml

# Confidence threshold sweep
python scripts/evaluate.py threshold --weights models/weights/detection.pt

# Test suite
make test
```

## The discipline that makes this credible

Say what is measured; say what is not. The gaps — no Jetson, no TensorRT, no
end-to-end OCR score, modest datasets — are all listed in `03_QA_PREP.md` with
an answer attached. One unbacked claim contaminates every backed one.
