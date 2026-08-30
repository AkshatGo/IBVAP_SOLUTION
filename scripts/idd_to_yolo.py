"""
IBVAP — IDD (Indian Driving Dataset) to YOLO format converter.

Converts IDD-Detection annotations into YOLO .txt label files, keeping only
the classes IBVAP acts on (person, bicycle, car, motorcycle, bus, truck),
and lays images/labels out into the train/val structure the training
pipeline expects.

Usage:
    python idd_to_yolo.py \
        --idd-root /path/to/idd_detection \
        --out-root data/detection \
        --val-split 0.15

Expected IDD-Detection input layout (as distributed by IDD):
    idd_detection/
        JPEGImages/*.jpg
        Annotations/*.xml      # PASCAL-VOC style XML, one per image

If your download uses a different annotation format (some IDD mirrors ship
JSON instead of XML), see `parse_json_annotation()` below as the alternate
path — swap the call in `main()` accordingly.
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# --- class mapping -----------------------------------------------------
# Left side: IDD's original label names (verify against your download's
# label list — IDD's taxonomy is broader than this; unmapped classes are
# dropped, not merged into "other", so they simply don't produce boxes).
IDD_TO_TARGET = {
    "person": "person",
    "rider": "person",          # riders are people; merge into person
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "vehicle fallback": "truck",  # IDD's catch-all for autos/tempos etc.
}

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}


def parse_voc_annotation(xml_path: Path):
    """Parse a PASCAL-VOC style IDD annotation XML.

    Returns: (image_width, image_height, list of (class_name, xmin, ymin, xmax, ymax))
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in IDD_TO_TARGET:
            continue  # drop classes we don't act on
        mapped = IDD_TO_TARGET[name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((mapped, xmin, ymin, xmax, ymax))

    return width, height, boxes


def parse_json_annotation(json_path: Path):
    """Alternate path if your IDD mirror ships JSON instead of XML.

    Expected shape (adjust keys to match your actual download —
    IDD's JSON export has varied across releases):
        {
          "imgWidth": 1920, "imgHeight": 1080,
          "objects": [{"label": "car", "bbox": [x1, y1, x2, y2]}, ...]
        }
    """
    import json

    with open(json_path) as f:
        data = json.load(f)

    width = data["imgWidth"]
    height = data["imgHeight"]

    boxes = []
    for obj in data.get("objects", []):
        name = obj["label"].strip().lower()
        if name not in IDD_TO_TARGET:
            continue
        mapped = IDD_TO_TARGET[name]
        x1, y1, x2, y2 = obj["bbox"]
        boxes.append((mapped, x1, y1, x2, y2))

    return width, height, boxes


def to_yolo_line(class_name: str, xmin, ymin, xmax, ymax, img_w, img_h) -> str:
    """Convert a pixel-space box into a normalized YOLO label line."""
    cls_idx = CLASS_TO_IDX[class_name]

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    # clamp to [0, 1] in case of annotation edge overshoot
    cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))

    return f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_dataset(idd_root: Path, out_root: Path, val_split: float, seed: int = 42):
    images_dir = idd_root / "JPEGImages"
    ann_dir = idd_root / "Annotations"

    if not images_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(
            f"Expected {images_dir} and {ann_dir} to exist. "
            "Check --idd-root points at the IDD-Detection folder, "
            "or adjust paths for your download's actual layout."
        )

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No .jpg images found under {images_dir}")

    random.seed(seed)
    random.shuffle(image_files)
    n_val = int(len(image_files) * val_split)
    val_set = set(image_files[:n_val])

    kept, skipped_no_boxes = 0, 0

    for img_path in image_files:
        xml_path = ann_dir / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue  # no annotation for this image, skip

        width, height, boxes = parse_voc_annotation(xml_path)
        if not boxes:
            skipped_no_boxes += 1
            continue  # image has zero boxes in our target classes — drop it,
            # keeping only background frames would need explicit intent, not
            # accidental inclusion.

        split = "val" if img_path in val_set else "train"

        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, out_img_dir / img_path.name)

        lines = [to_yolo_line(cls, x1, y1, x2, y2, width, height) for cls, x1, y1, x2, y2 in boxes]
        (out_lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")

        kept += 1

    print(f"Converted {kept} images ({n_val} val / {kept - n_val} train). "
          f"Skipped {skipped_no_boxes} images with no target-class boxes.")

    write_data_yaml(out_root)


def write_data_yaml(out_root: Path):
    yaml_content = (
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES)) + "\n"
    )
    (out_root / "data.yaml").write_text(yaml_content)
    print(f"Wrote {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Convert IDD-Detection to YOLO format")
    parser.add_argument("--idd-root", type=Path, required=True,
                         help="Path to the IDD-Detection folder (contains JPEGImages/, Annotations/)")
    parser.add_argument("--out-root", type=Path, default=Path("data/detection"),
                         help="Output folder, matches Phase 3 layout in the implementation plan")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_dataset(args.idd_root, args.out_root, args.val_split, args.seed)


if __name__ == "__main__":
    main()
