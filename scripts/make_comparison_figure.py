"""
IBVAP — build the stock-vs-fine-tuned comparison figure.

Produces the single image that makes the domain-adaptation argument
without a table: the same frame run through stock COCO-pretrained YOLOv8n
and through the IDD fine-tuned model, side by side.

The argument is not "our mAP is higher". It is that COCO has no
auto-rickshaw class at all, so stock weights must force one into whatever
COCO category fits worst — a truck, a bus, and in one real val frame, a
couch. Fine-tuning does not improve that label; it makes the label
possible.

Usage:
    python scripts/make_comparison_figure.py \
        --image data/detection/images/val/<stem>.jpg \
        --out SIH_first_pitch/assets/autorickshaw_comparison.png
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

if sys.platform == "linux" and multiprocessing.get_start_method(allow_none=True) != "fork":
    multiprocessing.set_start_method("fork", force=True)

import cv2
import numpy as np

# Anything the fine-tuned model names "autorickshaw" is the point of the
# figure, so it gets the one saturated colour; everything else stays muted
# and the eye goes where it should.
HIGHLIGHT = (60, 200, 255)    # amber, BGR
NEUTRAL = (170, 170, 170)
BAD = (70, 70, 235)           # red, for stock's wrong labels
PANEL_BG = (28, 34, 42)


def annotate(frame, model, conf, highlight_names, wrong_names=()):
    """Draw one model's detections onto a copy of the frame."""
    vis = frame.copy()
    result = model(frame, conf=conf, verbose=False)[0]
    counts = {}

    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

        if name in highlight_names:
            colour, thickness = HIGHLIGHT, 4
        elif name in wrong_names:
            colour, thickness = BAD, 4
        else:
            colour, thickness = NEUTRAL, 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)

        label = f"{name} {float(box.conf[0]):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 5, max(th, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    return vis, counts


def panel(image, title, subtitle, width, header=110):
    """Scale an annotated frame and stack a header strip above it."""
    scale = width / image.shape[1]
    body = cv2.resize(image, (width, int(image.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)

    strip = np.full((header, width, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(strip, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(strip, subtitle, (24, 84), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (150, 190, 220), 1, cv2.LINE_AA)
    return np.vstack([strip, body])


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Stock vs fine-tuned figure")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--stock", type=str, default="yolov8n.pt")
    parser.add_argument("--tuned", type=str, default="models/weights/detection.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=1100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read {args.image}")

    stock_model = YOLO(args.stock)
    tuned_model = YOLO(args.tuned)

    tuned_vis, tuned_counts = annotate(frame, tuned_model, args.conf,
                                       highlight_names={"autorickshaw"})
    # On the stock panel, flag the categories COCO forces an auto-rickshaw
    # into — those boxes are the argument.
    stock_vis, stock_counts = annotate(frame, stock_model, args.conf,
                                       highlight_names=set(),
                                       wrong_names={"couch", "truck", "bus", "bench"})

    n_auto = tuned_counts.get("autorickshaw", 0)
    wrong = ", ".join(f"{k} x{v}" for k, v in sorted(stock_counts.items())
                      if k in {"couch", "truck", "bus", "bench"}) or "no vehicle class"

    left = panel(stock_vis, "Stock YOLOv8n  ·  COCO-pretrained",
                 f"No autorickshaw class exists. Labelled: {wrong}", args.panel_width)
    right = panel(tuned_vis, "IBVAP  ·  fine-tuned on IDD",
                  f"autorickshaw detected x{n_auto}  ·  mAP@0.5 0.617 for this class",
                  args.panel_width)

    height = max(left.shape[0], right.shape[0])
    for name, img in (("left", left), ("right", right)):
        if img.shape[0] < height:
            pad = np.full((height - img.shape[0], img.shape[1], 3), PANEL_BG, np.uint8)
            if name == "left":
                left = np.vstack([left, pad])
            else:
                right = np.vstack([right, pad])

    divider = np.full((height, 8, 3), (90, 100, 115), dtype=np.uint8)
    figure = np.hstack([left, divider, right])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), figure)
    print(f"Wrote {args.out}  ({figure.shape[1]}x{figure.shape[0]})")
    print(f"  stock: {stock_counts}")
    print(f"  tuned: {tuned_counts}")


if __name__ == "__main__":
    main()
