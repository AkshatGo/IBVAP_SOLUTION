"""
IBVAP — model evaluation (ROADMAP §4, "Evaluation Metrics").

The point of this script is to stop a single headline number from hiding
the thing that actually matters. Two commands:

  detection — mAP@0.5 and mAP@0.5:0.95, reported for the whole val set and
              then broken out for the day and night subsets separately.
              A model can post a respectable overall mAP while being close
              to useless after dark; the split is what makes the night
              claim honest, and the day-minus-night gap is the number worth
              putting on a slide.

  anpr      — end-to-end plate accuracy (exact-match rate on the plate
              string) plus localization recall, reported per localizer.
              Run it once with the contour localizer and once with a
              trained one to isolate where a gain came from: localization
              should improve, OCR is unchanged EasyOCR either way.

Usage:
    python scripts/evaluate.py detection \
        --weights runs/detect/ibvap_detection/weights/best.pt \
        --data data/detection/data.yaml

    python scripts/evaluate.py anpr \
        --images-dir data/anpr/images/val \
        --ground-truth data/anpr/plates_val.json \
        --plate-model runs/detect/ibvap_plate/weights/best.pt

Night classification for the detection split uses filename prefixes by
default, matching what the preprocessing scripts emit: `exdark_` (real
low-light photographs) and `dark_` (synthetically darkened). Pass
--by-brightness to classify by mean luma instead, for a val set that
doesn't follow that convention.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NIGHT_PREFIXES = ("exdark_", "dark_")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --- detection ---------------------------------------------------------

def _is_night_by_name(image_path: Path) -> bool:
    return image_path.stem.startswith(NIGHT_PREFIXES)


def _is_night_by_brightness(image_path: Path, threshold: float) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return image is not None and float(np.mean(image)) < threshold


def _read_yaml_field(data_yaml: Path, key: str):
    """Minimal single-line YAML reader — avoids a pyyaml dependency."""
    for line in data_yaml.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _build_subset(image_paths, label_dir: Path, root: Path, names_block: str, tag: str):
    """Materialise a val-only YOLO dataset for a subset of images.

    Ultralytics evaluates whatever `val:` points at, so the cheapest way to
    score a subset is to hand it its own tiny dataset. Files are hardlinked
    where possible so this costs no meaningful disk.
    """
    subset_root = root / tag
    img_dir = subset_root / "images" / "val"
    lbl_dir = subset_root / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        try:
            (img_dir / image_path.name).hardlink_to(image_path)
        except (OSError, AttributeError):
            shutil.copy2(image_path, img_dir / image_path.name)
        if label_path.exists():
            try:
                (lbl_dir / label_path.name).hardlink_to(label_path)
            except (OSError, AttributeError):
                shutil.copy2(label_path, lbl_dir / label_path.name)

    yaml_path = subset_root / "data.yaml"
    yaml_path.write_text(
        f"path: {subset_root.resolve()}\n"
        f"train: images/val\n"   # unused; ultralytics wants the key present
        f"val: images/val\n\n"
        f"{names_block}\n"
    )
    return yaml_path


def _val(model, data_yaml: Path, imgsz: int, split_name: str):
    """Run validation and pull out the two mAP numbers."""
    from ultralytics import YOLO  # noqa: F401  (model is already a YOLO)

    metrics = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False,
                        split="val", plots=False)
    return {
        "split": split_name,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def evaluate_detection(args):
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Dataset config not found: {data_yaml}")

    dataset_root = Path(_read_yaml_field(data_yaml, "path") or data_yaml.parent)
    val_rel = _read_yaml_field(data_yaml, "val") or "images/val"
    img_dir = dataset_root / val_rel
    lbl_dir = Path(str(img_dir).replace("images", "labels", 1)) \
        if "images" in str(img_dir) else dataset_root / "labels" / "val"

    if not img_dir.is_dir():
        raise SystemExit(f"Val images not found at {img_dir}")

    images = [p for p in sorted(img_dir.iterdir()) if p.suffix in IMAGE_SUFFIXES]
    if not images:
        raise SystemExit(f"No val images in {img_dir}")

    if args.by_brightness:
        night = [p for p in images if _is_night_by_brightness(p, args.brightness_threshold)]
    else:
        night = [p for p in images if _is_night_by_name(p)]
    day = [p for p in images if p not in set(night)]

    print(f"Val set: {len(images)} images — {len(day)} day, {len(night)} night")
    if not night:
        print("  No night images identified; only the overall number will be "
              "meaningful. Check --by-brightness or your filename convention.")

    # Reproduce the class-name block verbatim so subset datasets keep the
    # same class indices as the parent.
    lines = data_yaml.read_text().splitlines()
    names_block = "\n".join(lines[lines.index("names:"):]) if "names:" in lines else "names:\n  0: object"

    model = YOLO(args.weights)
    rows = [_val(model, data_yaml, args.imgsz, "overall")]

    with tempfile.TemporaryDirectory(prefix="ibvap_eval_") as tmp:
        tmp_root = Path(tmp)
        for tag, subset in (("day", day), ("night", night)):
            if not subset:
                continue
            subset_yaml = _build_subset(subset, lbl_dir, tmp_root, names_block, tag)
            rows.append(_val(model, subset_yaml, args.imgsz, tag))

    print(f"\n{'split':<10}{'mAP@0.5':>12}{'mAP@0.5:0.95':>16}")
    print("-" * 38)
    for row in rows:
        print(f"{row['split']:<10}{row['map50']:>12.4f}{row['map50_95']:>16.4f}")

    by_split = {row["split"]: row for row in rows}
    if "day" in by_split and "night" in by_split:
        gap = by_split["day"]["map50"] - by_split["night"]["map50"]
        print(f"\nDay-night gap (mAP@0.5): {gap:+.4f}")
        print("Report this number rather than the overall mAP alone — it is "
              "the measured night-performance cost.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")
    return rows


# --- anpr --------------------------------------------------------------

def _write_gt_template(images_dir: Path, out_path: Path) -> int:
    """Write {filename: ""} for every image, preserving any existing labels.

    Transcribing 91 plates by hand is the real cost of this measurement, so
    a re-run must never overwrite work already done — existing entries are
    carried over and only new filenames are added blank.
    """
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    names = sorted(
        path.name for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    template = {name: existing.get(name, "") for name in names}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, indent=2, sort_keys=True))
    return sum(1 for value in template.values() if not value)


def _normalize_plate(text: str) -> str:
    """Compare plate strings ignoring case, spaces and hyphens."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def evaluate_anpr(args):
    from src.edge.anpr import ANPREngine

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        written = _write_gt_template(images_dir, gt_path)
        print(f"No ground truth at {gt_path} — wrote a template with "
              f"{written} empty entries.\n"
              f"Fill in the plate text for each image, then re-run this "
              f"command to get the end-to-end number.\n"
              f"Entries left empty are skipped, so a partial pass still "
              f"scores — it just scores fewer images.")
        return None

    ground_truth = json.loads(gt_path.read_text())
    # Accept either {"img.jpg": "MH12AB1234"} or {"img.jpg": {"plate": "..."}}
    truth = {
        k: (v if isinstance(v, str) else v.get("plate", ""))
        for k, v in ground_truth.items()
    }

    engine = ANPREngine(plate_model_path=args.plate_model).load()
    print(f"Localizer: {engine.localizer}")
    if args.plate_model and engine.localizer == "contour":
        print("  (requested model failed to load — results below are the "
              "contour fallback, not the trained localizer)")

    total = localized = exact = 0
    misses = []

    for name, expected in sorted(truth.items()):
        if not expected:
            # An unfilled template row. Counting it would silently deflate
            # exact-match toward zero and make a half-labelled run look like
            # a broken reader.
            continue
        image_path = images_dir / name
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        total += 1
        readings = engine.process_frame(image, frame_id=total)
        if readings:
            localized += 1

        best = max(readings, key=lambda r: r.confidence, default=None)
        predicted = best.plate_text if best else ""
        if _normalize_plate(predicted) == _normalize_plate(expected) and expected:
            exact += 1
        elif len(misses) < args.show_misses:
            misses.append((name, expected, predicted or "<none>"))

    if total == 0:
        raise SystemExit("No ground-truth images were readable — check paths.")

    print(f"\nEvaluated {total} images with the '{engine.localizer}' localizer")
    print(f"  Localization recall (>=1 candidate): {localized}/{total} = {localized / total:.1%}")
    print(f"  End-to-end exact match:              {exact}/{total} = {exact / total:.1%}")

    if misses:
        print(f"\nFirst {len(misses)} misses (expected -> predicted):")
        for name, expected, predicted in misses:
            print(f"  {name}: {expected} -> {predicted}")

    result = {
        "localizer": engine.localizer,
        "images": total,
        "localization_recall": localized / total,
        "exact_match": exact / total,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")
    return result


# --- confidence threshold sweep ----------------------------------------

def sweep_threshold(args):
    """Re-validate the pipeline's detection cutoff against a new model.

    src/config.py ships confidence_threshold=0.45, chosen against stock
    YOLOv8n. Fine-tuning shifts the confidence distribution, so that number
    is an assumption, not a constant — this sweep re-derives it. Pick the
    threshold by what the deployment should optimise: recall matters more
    than precision at a border, so a lower cutoff than peak-F1 is a
    defensible choice as long as it is a choice.
    """
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Dataset config not found: {data_yaml}")

    model = YOLO(args.weights)
    rows = []
    for conf in args.thresholds:
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, conf=conf,
                            verbose=False, plots=False)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({"conf": conf, "precision": precision, "recall": recall,
                     "f1": f1, "map50": float(metrics.box.map50)})

    print(f"\n{'conf':>8}{'precision':>12}{'recall':>10}{'F1':>10}{'mAP@0.5':>12}")
    print("-" * 52)
    for row in rows:
        print(f"{row['conf']:>8.2f}{row['precision']:>12.4f}{row['recall']:>10.4f}"
              f"{row['f1']:>10.4f}{row['map50']:>12.4f}")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nPeak F1 at conf={best['conf']:.2f} (F1={best['f1']:.4f}).")
    print("Current src/config.py setting: confidence_threshold=0.45")
    if abs(best["conf"] - 0.45) >= 0.1:
        print("  -> Peak F1 has moved away from 0.45; update the config or "
              "state why you're keeping it.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.out}")
    return rows


# --- localizer A/B ------------------------------------------------------

def _iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_gt(label_path: Path, img_w: int, img_h: int):
    """Read YOLO labels back into absolute (x1, y1, x2, y2) boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((
            (cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h,
        ))
    return boxes


def score_localizer(engine, images, labels_dir: Path, iou_threshold: float):
    """Greedy IoU matching of predictions to ground truth.

    Each ground-truth box may be claimed once. Unmatched predictions are
    false positives, unmatched ground truth false negatives — the same
    accounting a detection mAP uses, at a single IoU threshold.
    """
    import cv2

    tp = fp = fn = 0
    images_with_gt = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        truth = _load_gt(labels_dir / f"{image_path.stem}.txt", img_w, img_h)
        if not truth:
            continue
        images_with_gt += 1

        preds = [(x, y, x + w, y + h)
                 for (x, y, w, h) in engine.detect_plate_region(image)]

        claimed = set()
        for pred in preds:
            best_iou, best_idx = 0.0, None
            for idx, gt in enumerate(truth):
                if idx in claimed:
                    continue
                value = _iou(pred, gt)
                if value > best_iou:
                    best_iou, best_idx = value, idx
            if best_idx is not None and best_iou >= iou_threshold:
                claimed.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += len(truth) - len(claimed)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"images": images_with_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def compare_localizers(args):
    """Score the contour baseline and the trained localizer side by side.

    Without the baseline a trained-model number cannot be attributed: the
    whole claim is that replacing contour localization improved things, and
    that requires both halves measured on the same held-out images.
    """
    from src.edge.anpr import ANPREngine

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix in IMAGE_SUFFIXES]
    if not images:
        raise SystemExit(f"No images in {images_dir}")

    rows = []

    # Baseline: no model at all — the classical contour detector.
    baseline = ANPREngine()
    print(f"Scoring '{baseline.localizer}' baseline on {len(images)} images...")
    rows.append(("contour (baseline)", score_localizer(baseline, images, labels_dir,
                                                       args.iou)))

    if args.plate_model:
        trained = ANPREngine(plate_model_path=args.plate_model,
                             plate_model_confidence=args.conf).load_localizer()
        if trained.localizer != "yolo":
            raise SystemExit(f"Could not load {args.plate_model}")
        print(f"Scoring trained localizer ({args.plate_model})...")
        rows.append(("yolo (trained)", score_localizer(trained, images, labels_dir,
                                                       args.iou)))

    print(f"\nLocalization at IoU >= {args.iou}")
    print(f"{'localizer':<22}{'precision':>11}{'recall':>9}{'F1':>9}"
          f"{'TP':>7}{'FP':>7}{'FN':>6}")
    print("-" * 71)
    for name, r in rows:
        print(f"{name:<22}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>9.3f}"
              f"{r['tp']:>7}{r['fp']:>7}{r['fn']:>6}")

    if len(rows) == 2:
        base, trained_row = rows[0][1], rows[1][1]
        print(f"\nDelta (trained - baseline):")
        print(f"  precision {trained_row['precision'] - base['precision']:+.3f}")
        print(f"  recall    {trained_row['recall'] - base['recall']:+.3f}")
        print(f"  F1        {trained_row['f1'] - base['f1']:+.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {name: r for name, r in rows}, indent=2))
        print(f"\nWrote {args.out}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate IBVAP's fine-tuned models")
    sub = parser.add_subparsers(dest="task", required=True)

    p_det = sub.add_parser("detection", help="mAP with a day/night breakout")
    p_det.add_argument("--weights", type=str, required=True)
    p_det.add_argument("--data", type=str, default="data/detection/data.yaml")
    p_det.add_argument("--imgsz", type=int, default=640)
    p_det.add_argument("--by-brightness", action="store_true",
                       help="Classify night by mean luma instead of filename prefix")
    p_det.add_argument("--brightness-threshold", type=float, default=60.0)
    p_det.add_argument("--out", type=str, default=None, help="Write results as JSON")

    p_anpr = sub.add_parser("anpr", help="End-to-end plate accuracy")
    p_anpr.add_argument("--images-dir", type=str, required=True)
    p_anpr.add_argument("--ground-truth", type=str, required=True,
                        help="JSON mapping filename -> plate string. If the "
                             "file does not exist it is created as a blank "
                             "template to fill in, and nothing is scored.")
    p_anpr.add_argument("--plate-model", type=str, default=None,
                        help="Trained localizer; omit to score the contour fallback")
    p_anpr.add_argument("--show-misses", type=int, default=10)
    p_anpr.add_argument("--out", type=str, default=None, help="Write results as JSON")

    p_thresh = sub.add_parser("threshold",
                              help="Sweep the detection confidence cutoff")
    p_thresh.add_argument("--weights", type=str, required=True)
    p_thresh.add_argument("--data", type=str, default="data/detection/data.yaml")
    p_thresh.add_argument("--imgsz", type=int, default=640)
    p_thresh.add_argument("--thresholds", type=float, nargs="+",
                          default=[0.25, 0.35, 0.45, 0.55, 0.65])
    p_thresh.add_argument("--out", type=str, default=None, help="Write results as JSON")

    p_loc = sub.add_parser("localizer",
                           help="Contour baseline vs trained plate localizer")
    p_loc.add_argument("--images-dir", type=str, default="data/anpr/images/val")
    p_loc.add_argument("--labels-dir", type=str, default="data/anpr/labels/val")
    p_loc.add_argument("--plate-model", type=str, default=None,
                       help="Trained localizer; omit to score only the baseline")
    p_loc.add_argument("--iou", type=float, default=0.5)
    p_loc.add_argument("--conf", type=float, default=0.35)
    p_loc.add_argument("--out", type=str, default=None)

    args = parser.parse_args()
    if args.task == "localizer":
        compare_localizers(args)
    elif args.task == "detection":
        evaluate_detection(args)
    elif args.task == "threshold":
        sweep_threshold(args)
    else:
        evaluate_anpr(args)


if __name__ == "__main__":
    main()
