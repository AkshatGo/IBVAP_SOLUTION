"""
IBVAP — ExDark to YOLO converter + synthetic low-light augmentation.

Implements the low-light half of the detection dataset (ROADMAP §3.4): one
model that generalises across the full lighting spectrum, rather than
separate day and night models.

Two modes, usually run in this order:

1. `convert` — turn the ExDark dataset into YOLO labels, mapping ExDark's
   12 classes down to the 5 IBVAP classes it actually covers. ExDark has no
   truck class, so a truck-heavy night scene still leans on IDD.

2. `darken` — take an already-converted daylight YOLO dataset (the IDD
   output of `idd_to_yolo.py`) and bake synthetic low-light copies of a
   fraction of its *train* images into the same tree. Geometry is untouched
   so labels are copied verbatim.

Usage:
    # ExDark -> YOLO, merged into the existing detection dataset
    python exdark_to_yolo.py convert \
        --exdark-root /path/to/ExDark \
        --out-root data/detection \
        --val-split 0.20

    # Synthesise night copies from the daylight IDD images already there
    python exdark_to_yolo.py darken \
        --out-root data/detection \
        --fraction 0.30

Expected ExDark input layout (as distributed):
    ExDark/
        ExDark/<ClassName>/*.jpg          # images, one folder per class
        ExDark_Annno/<ClassName>/*.txt    # annotations, one .txt per image

ExDark annotation lines are `label left top width height ...` in absolute
pixels, after a leading `%`-prefixed comment line.

IMPORTANT — the validation split stays genuinely dark. Synthetic darkening
is written to `train` only, so the reported night mAP is measured on real
low-light photographs and not inflated by easy synthetic examples.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# --- class mapping -----------------------------------------------------
# ExDark folder/label names on the left. Classes ExDark has but IBVAP does
# not act on (Boat, Bottle, Cat, Chair, Cup, Dog, Table) are dropped, not
# merged, so they produce no boxes.
EXDARK_TO_TARGET = {
    "bicycle": "bicycle",
    "bus": "bus",
    "car": "car",
    "motorbike": "motorcycle",
    "people": "person",
}

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --- ExDark -> YOLO ----------------------------------------------------

def parse_exdark_annotation(ann_path: Path, img_w: int, img_h: int):
    """Parse one ExDark .txt annotation into YOLO label lines.

    ExDark format, one object per line, absolute pixels:
        <label> <left> <top> <width> <height> <...ignored...>
    The first line is a `%`-prefixed header and is skipped.

    Returns a list of "cls cx cy w h" strings, normalised to [0, 1].
    Objects whose label is not in EXDARK_TO_TARGET are dropped.
    """
    lines = []
    for raw in ann_path.read_text(errors="ignore").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("%"):
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue

        label = parts[0].strip().lower()
        target = EXDARK_TO_TARGET.get(label)
        if target is None:
            continue

        try:
            left, top, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue

        # Absolute (left, top, w, h) -> normalised (cx, cy, w, h), clamped
        # to the frame in case an annotation runs past the image edge.
        left = max(0.0, min(left, img_w))
        top = max(0.0, min(top, img_h))
        w = min(w, img_w - left)
        h = min(h, img_h - top)
        if w <= 0 or h <= 0:
            continue

        cx = (left + w / 2) / img_w
        cy = (top + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        lines.append(f"{CLASS_TO_IDX[target]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def _find_annotation(ann_root: Path, class_dir: str, image_path: Path):
    """Locate the annotation for an image.

    ExDark names annotations `<image filename>.txt`, i.e. the image suffix is
    kept (`img_001.jpg.txt`). Some mirrors drop it (`img_001.txt`), so try
    both before giving up.
    """
    candidates = [
        ann_root / class_dir / f"{image_path.name}.txt",
        ann_root / class_dir / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def convert_exdark(exdark_root: Path, out_root: Path, val_split: float, seed: int = 42):
    """Convert ExDark into the YOLO tree at out_root, merging with what's there."""
    # Tolerate both `ExDark/ExDark/...` (as shipped) and a flattened layout.
    img_root = exdark_root / "ExDark"
    if not img_root.is_dir():
        img_root = exdark_root
    ann_root = exdark_root / "ExDark_Annno"
    if not ann_root.is_dir():
        ann_root = exdark_root / "ExDark_Anno"
    if not ann_root.is_dir():
        raise SystemExit(
            f"Could not find ExDark annotations under {exdark_root} "
            "(looked for ExDark_Annno/ and ExDark_Anno/)"
        )

    pairs = []
    skipped_no_ann = 0
    for class_dir in sorted(p.name for p in img_root.iterdir() if p.is_dir()):
        if class_dir.lower() not in EXDARK_TO_TARGET:
            continue  # class IBVAP does not act on
        for image_path in sorted((img_root / class_dir).iterdir()):
            if image_path.suffix not in IMAGE_SUFFIXES:
                continue
            ann_path = _find_annotation(ann_root, class_dir, image_path)
            if ann_path is None:
                skipped_no_ann += 1
                continue
            pairs.append((image_path, ann_path))

    if not pairs:
        raise SystemExit(f"No annotated ExDark images found under {img_root}")

    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_split)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    counts = {"train": 0, "val": 0}
    empty = 0
    for split, split_pairs in splits.items():
        img_out = out_root / "images" / split
        lbl_out = out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for image_path, ann_path in split_pairs:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            lines = parse_exdark_annotation(ann_path, img_w, img_h)
            if not lines:
                empty += 1
                continue  # no target-class objects; nothing to learn from

            # Prefix guards against filename collisions with the IDD images
            # already sitting in this tree.
            stem = f"exdark_{image_path.stem}"
            shutil.copy2(image_path, img_out / f"{stem}{image_path.suffix}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

    print(f"ExDark -> YOLO: {counts['train']} train, {counts['val']} val images")
    if skipped_no_ann:
        print(f"  skipped {skipped_no_ann} images with no annotation file")
    if empty:
        print(f"  skipped {empty} images with no target-class objects")
    return counts


# --- synthetic low-light -----------------------------------------------

def darken_image(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply synthetic low-light degradation to a daylight image.

    Three effects, in the order a real sensor produces them:
      1. gamma reduction (0.3-0.6)  — the underexposure itself
      2. contrast reduction          — the flattening that comes with it
      3. Gaussian sensor noise       — what the sensor adds at high ISO

    Geometry is untouched, so the YOLO labels remain valid unchanged.
    """
    gamma = rng.uniform(0.3, 0.6)
    # Build the gamma LUT once per image rather than per pixel.
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    out = cv2.LUT(image, lut)

    contrast = rng.uniform(0.6, 0.85)
    out = cv2.convertScaleAbs(out, alpha=contrast, beta=0)

    sigma = rng.uniform(4.0, 12.0)
    out = out.astype(np.int16) + np.random.normal(0, sigma, out.shape).astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def darken_split(out_root: Path, fraction: float, seed: int = 42):
    """Bake synthetic night copies of a fraction of the train split.

    Writes into `train` only — `val` is left alone so night metrics are
    measured on genuinely dark images (ROADMAP §3.4).
    """
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    if not img_dir.is_dir():
        raise SystemExit(f"No train images at {img_dir} — run idd_to_yolo.py first")

    # Only darken daylight source images; skip real ExDark ones (already
    # dark) and anything this script produced on a previous run.
    sources = [
        p for p in sorted(img_dir.iterdir())
        if p.suffix in IMAGE_SUFFIXES
        and not p.stem.startswith("exdark_")
        and not p.stem.startswith("dark_")
    ]
    if not sources:
        raise SystemExit(f"No daylight source images found in {img_dir}")

    rng = random.Random(seed)
    np.random.seed(seed)
    picked = rng.sample(sources, max(1, int(len(sources) * fraction)))

    written = 0
    for image_path in picked:
        label_path = lbl_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        stem = f"dark_{image_path.stem}"
        cv2.imwrite(str(img_dir / f"{stem}{image_path.suffix}"), darken_image(image, rng))
        shutil.copy2(label_path, lbl_dir / f"{stem}.txt")
        written += 1

    print(f"Synthetic low-light: {written} darkened copies added to train "
          f"(from {len(sources)} daylight images, fraction={fraction})")
    return written


# --- data.yaml ---------------------------------------------------------

def write_data_yaml(out_root: Path):
    """Rewrite data.yaml so it reflects the merged dataset."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES))
    content = (
        f"# IBVAP detection dataset (IDD + ExDark + synthetic low-light)\n"
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n{names}\n"
    )
    (out_root / "data.yaml").write_text(content)
    print(f"Wrote {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ExDark to YOLO format and synthesise low-light training data"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_convert = sub.add_parser("convert", help="Convert ExDark into the YOLO tree")
    p_convert.add_argument("--exdark-root", type=Path, required=True,
                           help="Root of the extracted ExDark dataset")
    p_convert.add_argument("--out-root", type=Path, default=Path("data/detection"),
                           help="Detection dataset root (merges with IDD output)")
    p_convert.add_argument("--val-split", type=float, default=0.20)
    p_convert.add_argument("--seed", type=int, default=42)

    p_darken = sub.add_parser("darken", help="Bake synthetic night copies of train images")
    p_darken.add_argument("--out-root", type=Path, default=Path("data/detection"))
    p_darken.add_argument("--fraction", type=float, default=0.30,
                          help="Fraction of daylight train images to darken")
    p_darken.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "convert":
        convert_exdark(args.exdark_root, args.out_root, args.val_split, args.seed)
    elif args.mode == "darken":
        darken_split(args.out_root, args.fraction, args.seed)

    write_data_yaml(args.out_root)


if __name__ == "__main__":
    main()
