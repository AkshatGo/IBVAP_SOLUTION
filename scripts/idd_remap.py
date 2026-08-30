"""
IBVAP — remap the already-YOLO IDD-Detection mirror to IBVAP's classes.

`idd_to_yolo.py` converts IDD's original PASCAL-VOC annotations. The Kaggle
mirror (`redzapdos123/indian-driving-dataset-detections-yolov11`) ships the
same data already in YOLO format with a 15-class taxonomy, so it needs
remapping rather than parsing.

The class list is where the domain argument lives. IDD's taxonomy contains
**autorickshaw**, which COCO does not have at all — a COCO-pretrained model
cannot detect one as anything but a mislabelled car or truck. Keeping it as
its own class is the concrete, demonstrable case for fine-tuning on Indian
road data, so IBVAP's six classes become seven here.

    person, bicycle, car, motorcycle, bus, truck, autorickshaw

`rider` folds into person (a rider is a person). `vehicle fallback`,
`caravan` and `trailer` fold into truck — IDD's catch-alls for large
vehicles that don't fit a named class. Classes IBVAP does not act on
(animal, traffic light, traffic sign, train) are dropped: an image keeping
only dropped classes contributes no boxes and is skipped.

Usage:
    # Full dataset
    python scripts/idd_remap.py \
        --src data/raw/idd-yolo/IDDDetectionsYOLODataset \
        --out data/detection

    # Subset, for a run that finishes in hours rather than overnight
    python scripts/idd_remap.py --src ... --out data/detection \
        --max-train 8000 --max-val 1000
"""

import argparse
import random
import shutil
from pathlib import Path

# IDD mirror's class order, from its data.yaml.
IDD_CLASSES = [
    "animal", "autorickshaw", "bicycle", "bus", "car", "caravan",
    "motorcycle", "person", "rider", "traffic light", "traffic sign",
    "trailer", "train", "truck", "vehicle fallback",
]

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck",
                  "autorickshaw"]
CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}

# Source class name -> target class name. Absent = dropped.
REMAP = {
    "person": "person",
    "rider": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "caravan": "truck",
    "trailer": "truck",
    "vehicle fallback": "truck",
    "autorickshaw": "autorickshaw",
}

# Precomputed source-index -> target-index, so the hot loop is a dict hit.
INDEX_MAP = {
    i: CLASS_TO_IDX[REMAP[name]]
    for i, name in enumerate(IDD_CLASSES) if name in REMAP
}


def remap_label(label_path: Path):
    """Rewrite one YOLO label file's class indices. Returns kept lines."""
    lines = []
    for raw in label_path.read_text(errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            src_idx = int(parts[0])
        except ValueError:
            continue
        target = INDEX_MAP.get(src_idx)
        if target is None:
            continue  # class IBVAP does not act on
        lines.append(" ".join([str(target)] + parts[1:5]))
    return lines


def convert_split(src_root: Path, out_root: Path, split: str,
                  limit: int | None, seed: int):
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    if not src_images.is_dir():
        print(f"  {split}: no images at {src_images}, skipping")
        return 0

    images = sorted(p for p in src_images.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        return 0

    # Sample before doing any work — reading 33k label files to then discard
    # most of them wastes minutes.
    if limit and len(images) > limit:
        images = random.Random(seed).sample(images, limit)

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept = empty = 0
    for image_path in images:
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = remap_label(label_path)
        if not lines:
            empty += 1
            continue  # nothing IBVAP acts on in this frame

        # Hardlink rather than copy: 20GB of images duplicated on disk for
        # no reason otherwise.
        destination = out_images / image_path.name
        if not destination.exists():
            try:
                destination.hardlink_to(image_path)
            except (OSError, AttributeError):
                shutil.copy2(image_path, destination)
        (out_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")
        kept += 1

    print(f"  {split}: {kept} images ({empty} skipped — no target classes)")
    return kept


def write_data_yaml(out_root: Path):
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES))
    (out_root / "data.yaml").write_text(
        f"# IBVAP detection dataset — IDD-Detection remapped to IBVAP classes\n"
        f"# autorickshaw is kept as its own class: COCO has no equivalent,\n"
        f"# which is the concrete case for fine-tuning on Indian road data.\n"
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names}\n"
    )
    print(f"Wrote {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Remap the YOLO-format IDD mirror to IBVAP's classes"
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="IDDDetectionsYOLODataset root (has train/ val/ test/)")
    parser.add_argument("--out", type=Path, default=Path("data/detection"))
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap training images — use for a subset run")
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source not found: {args.src}")

    print(f"Remapping {len(IDD_CLASSES)} IDD classes -> "
          f"{len(TARGET_CLASSES)} IBVAP classes")
    total = 0
    total += convert_split(args.src, args.out, "train", args.max_train, args.seed)
    total += convert_split(args.src, args.out, "val", args.max_val, args.seed)
    if total == 0:
        raise SystemExit("No images converted — check --src layout")
    write_data_yaml(args.out)


if __name__ == "__main__":
    main()
