"""
IBVAP — plate datasets to YOLO format (ROADMAP §3.3 step 1).

Converts the two ANPR source datasets into one single-class (`plate`) YOLO
dataset, so a single localizer fine-tune can consume both. This is the
missing first half of the ANPR pipeline: `anpr_augmentation.py` augments a
YOLO-format plate dataset, but nothing produced one until now.

Two annotation formats, because the two datasets ship differently:

  voc   — PASCAL-VOC XML, one .xml per image, `<bndbox>` in absolute
          pixels. This is what the common Kaggle plate datasets use
          (e.g. andrewmvd/car-plate-detection).

  ufpr  — UFPR-ALPR's per-image .txt, which carries a
          `position_plate: x y w h` line in absolute pixels alongside the
          plate string and other metadata.

Usage:
    # Kaggle-style VOC dataset
    python scripts/plates_to_yolo.py \
        --root data/raw/car-plate-detection --format voc \
        --out-root data/anpr --val-split 0.20

    # UFPR-ALPR, merged into the same output tree
    python scripts/plates_to_yolo.py \
        --root data/raw/UFPR-ALPR --format ufpr --out-root data/anpr

Both write into one tree; run them in sequence to merge. Filenames are
prefixed per source so the two datasets cannot collide.

The 80/20 split is deliberate and strict (ROADMAP §3.3 step 3): a
single-class localizer on a few thousand images overfits easily, so the
held-out set has to be large enough for the val curve to mean something.

`--ground-truth-out` additionally writes the plate strings UFPR ships as a
JSON map, which is the input `scripts/evaluate.py anpr` needs to score
end-to-end OCR accuracy. VOC datasets rarely carry the plate text, so that
file is only written when strings are actually found.
"""

import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Single class: the localizer's only job is finding plates. Reading them
# stays with EasyOCR, which is not fine-tuned (ROADMAP §3.3 step 5).
CLASS_NAMES = ["plate"]

# VOC label names that mean "plate" across the various Kaggle mirrors.
PLATE_LABELS = {
    "licence", "license", "license_plate", "licence_plate", "plate",
    "number_plate", "numberplate", "license-plate",
}


def to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h) -> str:
    """Absolute corner box -> normalised YOLO line, class 0."""
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# --- PASCAL-VOC XML ----------------------------------------------------

def parse_voc(xml_path: Path):
    """Return (width, height, [(xmin, ymin, xmax, ymax), ...]).

    Every object is treated as a plate unless its name clearly isn't one —
    these datasets are single-purpose, and mirrors disagree on whether the
    label is "licence", "license_plate" or "plate".
    """
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        return None, None, []
    width = int(float(size.findtext("width", 0)))
    height = int(float(size.findtext("height", 0)))
    if width <= 0 or height <= 0:
        return None, None, []

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower().replace(" ", "_")
        if name and name not in PLATE_LABELS:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))

    return width, height, boxes


def collect_voc(root: Path):
    """Pair each image with its XML annotation."""
    image_dirs = [root / "images", root / "JPEGImages", root]
    ann_dirs = [root / "annotations", root / "Annotations", root]

    images_dir = next((d for d in image_dirs if d.is_dir()), None)
    ann_dir = next((d for d in ann_dirs if d.is_dir()), None)
    if images_dir is None or ann_dir is None:
        raise SystemExit(f"Could not find image/annotation folders under {root}")

    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix not in IMAGE_SUFFIXES:
            continue
        xml_path = ann_dir / f"{image_path.stem}.xml"
        if xml_path.exists():
            pairs.append((image_path, xml_path))
    return pairs


# --- UFPR-ALPR ---------------------------------------------------------

_POSITION_RE = re.compile(r"position_plate:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", re.I)
_PLATE_RE = re.compile(r"^plate:\s*(\S+)", re.I | re.M)


def parse_ufpr(txt_path: Path):
    """Return ([(xmin, ymin, xmax, ymax), ...], plate_string or None).

    UFPR gives `position_plate: x y w h` in absolute pixels — corner plus
    size, not corner plus corner, so it needs converting before it can be
    treated like a VOC box.
    """
    text = txt_path.read_text(errors="ignore")

    boxes = []
    for match in _POSITION_RE.finditer(text):
        x, y, w, h = (int(v) for v in match.groups())
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))

    plate_match = _PLATE_RE.search(text)
    plate = plate_match.group(1).strip().upper() if plate_match else None
    return boxes, plate


def collect_ufpr(root: Path):
    """UFPR ships training/validation/testing trees of per-track folders."""
    pairs = []
    for image_path in sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg")):
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((image_path, txt_path))
    return pairs


# --- conversion --------------------------------------------------------

def convert(root: Path, fmt: str, out_root: Path, val_split: float,
            seed: int, ground_truth_out: Path | None):
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    pairs = collect_voc(root) if fmt == "voc" else collect_ufpr(root)
    if not pairs:
        raise SystemExit(
            f"No annotated images found under {root} for format '{fmt}'. "
            "Check --root points at the extracted dataset."
        )

    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_split)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    counts = {"train": 0, "val": 0}
    empty = 0
    plate_strings = {}

    for split, split_pairs in splits.items():
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for image_path, ann_path in split_pairs:
            plate = None
            if fmt == "voc":
                width, height, boxes = parse_voc(ann_path)
            else:
                boxes, plate = parse_ufpr(ann_path)
                # UFPR's annotation carries no image size, so read it.
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]

            if not boxes or not width or not height:
                empty += 1
                continue

            # Prefix by source so the two datasets can share one tree.
            stem = f"{fmt}_{image_path.stem}"
            filename = f"{stem}{image_path.suffix}"
            shutil.copy2(image_path, img_out / filename)

            lines = [to_yolo_line(*box, width, height) for box in boxes]
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

            if plate and split == "val":
                # Only val strings matter: they are what evaluate.py scores.
                plate_strings[filename] = plate

    print(f"{fmt.upper()} -> YOLO: {counts['train']} train, {counts['val']} val images")
    if empty:
        print(f"  skipped {empty} images with no usable plate box")

    if plate_strings and ground_truth_out:
        existing = {}
        if ground_truth_out.exists():
            existing = json.loads(ground_truth_out.read_text())
        existing.update(plate_strings)
        ground_truth_out.write_text(json.dumps(existing, indent=2, sort_keys=True))
        print(f"  wrote {len(plate_strings)} val plate strings to {ground_truth_out}")
    elif fmt == "ufpr" and not plate_strings:
        print("  no plate strings found — end-to-end OCR scoring will need "
              "a ground-truth file built by hand")

    return counts


def write_data_yaml(out_root: Path):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (out_root / "data.yaml").write_text(
        f"# IBVAP plate localizer dataset (single class)\n"
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names}\n"
    )
    print(f"Wrote {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert plate datasets to single-class YOLO format"
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Extracted dataset root")
    parser.add_argument("--format", choices=["voc", "ufpr"], required=True,
                        help="voc: PASCAL-VOC XML (Kaggle plate datasets); "
                             "ufpr: UFPR-ALPR per-image .txt")
    parser.add_argument("--out-root", type=Path, default=Path("data/anpr"))
    parser.add_argument("--val-split", type=float, default=0.20,
                        help="Kept strict at 0.20 — a single-class localizer "
                             "overfits easily (ROADMAP §3.3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground-truth-out", type=Path,
                        default=Path("data/anpr/plates_val.json"),
                        help="Where to write val plate strings for "
                             "scripts/evaluate.py anpr")
    args = parser.parse_args()

    convert(args.root, args.format, args.out_root, args.val_split,
            args.seed, args.ground_truth_out)
    write_data_yaml(args.out_root)


if __name__ == "__main__":
    main()
