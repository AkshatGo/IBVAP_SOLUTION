"""
IBVAP — ANPR plate-detection augmentation pipeline.

Source datasets (Indian Number Plates, UFPR-ALPR) are mostly clean, frontal,
well-lit shots. Real border-road CCTV plates are angled, motion-blurred,
low-resolution, and shot at dusk/night. This pipeline applies exactly those
distortions at training time so the fine-tuned plate localizer generalizes
to what the edge cameras will actually see, per Phase 3.3 of the
implementation plan.

Two entry points:
  - get_train_transform() / get_val_transform(): Albumentations pipelines
    for use inside a YOLO/torch training loop (bbox-aware).
  - augment_and_export(): a standalone script mode that reads a YOLO-format
    dataset, generates N augmented copies per image, and writes them out —
    useful if you want to bake augmentation into a static dataset rather
    than doing it on-the-fly during training (e.g. to sanity-check outputs
    visually before a long training run).

Usage (standalone export mode):
    python anpr_augmentation.py \
        --images-dir data/anpr/images/train \
        --labels-dir data/anpr/labels/train \
        --out-dir data/anpr_augmented/train \
        --copies-per-image 3
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def get_train_transform(img_size: int = 640) -> A.Compose:
    """Augmentation pipeline for training the plate localizer.

    Order matters: geometric transforms (perspective, rotation) run before
    quality-degradation transforms (blur, noise, resolution), so blur isn't
    warped afterward in a way that looks unnatural.
    """
    return A.Compose(
        [
            # --- geometric: simulate non-frontal, roadside camera angles ---
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),

            # --- motion: simulate a passing vehicle, not a parked one ---
            A.MotionBlur(blur_limit=(3, 9), p=0.4),

            # --- lighting: day/dusk/artificial-light variance at a BOP ---
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),

            # --- sensor realism: cheap CCTV, not a DSLR ---
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(p=0.2),

            # --- resolution: simulate a distant/low-res CCTV plate ---
            A.Downscale(scale_min=0.5, scale_max=0.7, interpolation=cv2.INTER_LINEAR, p=0.4),

            # --- final resize/pad to model input size ---
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def get_val_transform(img_size: int = 640) -> A.Compose:
    """Validation gets only resize/pad — no synthetic distortion.

    Keeping val clean is what makes any reported accuracy delta honest
    (see the "genuinely dark-only held-out val" note in the implementation
    plan for the same reasoning applied to night detection).
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def _read_yolo_labels(label_path: Path):
    """Read a YOLO .txt label file -> (bboxes, class_labels)."""
    bboxes, class_labels = [], []
    if not label_path.exists():
        return bboxes, class_labels
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        bboxes.append([float(cx), float(cy), float(w), float(h)])
        class_labels.append(int(cls))
    return bboxes, class_labels


def _write_yolo_labels(label_path: Path, bboxes, class_labels):
    lines = [
        f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, (cx, cy, w, h) in zip(class_labels, bboxes)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def augment_and_export(
    images_dir: Path, labels_dir: Path, out_dir: Path,
    copies_per_image: int = 3, img_size: int = 640, seed: int = 42,
):
    """Generate augmented copies of a YOLO-format plate dataset on disk."""
    random.seed(seed)
    np.random.seed(seed)

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    transform = get_train_transform(img_size)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found under {images_dir}")

    written = 0
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = labels_dir / (img_path.stem + ".txt")
        bboxes, class_labels = _read_yolo_labels(label_path)
        if not bboxes:
            continue  # skip images with no plate annotation

        for copy_idx in range(copies_per_image):
            try:
                result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                # a small fraction of augmentations can drop all boxes below
                # min_visibility after aggressive perspective warp — skip
                # that single sample rather than failing the whole run
                print(f"Skipped {img_path.name} copy {copy_idx}: {e}")
                continue

            if not result["bboxes"]:
                continue  # augmentation pushed the plate fully out of frame

            out_name = f"{img_path.stem}_aug{copy_idx}"
            out_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_images / f"{out_name}.jpg"), out_img)
            _write_yolo_labels(out_labels / f"{out_name}.txt", result["bboxes"], result["class_labels"])
            written += 1

    print(f"Wrote {written} augmented image/label pairs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment a YOLO-format ANPR plate dataset")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copies-per-image", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    augment_and_export(
        args.images_dir, args.labels_dir, args.out_dir,
        copies_per_image=args.copies_per_image, img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
