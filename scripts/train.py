"""
IBVAP — YOLOv8 fine-tuning runner (ROADMAP §4).

Two models, one script:

  detection — person/vehicle detection fine-tuned on the merged
              IDD + ExDark + synthetic-low-light dataset. Uses a
              freeze-backbone warm-up: the first N epochs train only the
              head, then the whole network unfreezes. On a few thousand
              images that converges faster and overfits less than
              unfreezing everything from epoch 0.

  plate     — single-class plate localiser, to replace the classical
              contour-based localisation in src/edge/anpr.py. One class
              converges fast, so the risk here is overfitting, not
              underfitting — keep the val split strict and watch the gap.

Both stay on yolov8n: the edge-deployment story depends on nano-sized
weights. Moving to yolov8s/m is a hardware-budget decision, not a
free accuracy win.

Usage:
    python scripts/train.py detection --data data/detection/data.yaml --epochs 80
    python scripts/train.py plate     --data data/anpr/data.yaml      --epochs 50

Outputs land in runs/detect/<name>/weights/{best,last}.pt. Point the edge
pipeline at one with:
    IBVAP_DETECTION_MODEL=runs/detect/ibvap_detection/weights/best.pt
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

# Python 3.14 made "forkserver" the default start method on Linux, which
# PyTorch's DataLoader workers do not survive — they die during forkserver
# authentication with ConnectionResetError before the first batch. Force
# "fork" before torch is imported anywhere.
if sys.platform == "linux" and multiprocessing.get_start_method(allow_none=True) != "fork":
    multiprocessing.set_start_method("fork", force=True)


def _resolve_device(requested: str) -> str:
    """Pick a training device, and say plainly what was picked and why."""
    import torch

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Device: cuda:0 ({name})")
        return "0"

    print("Device: cpu — no CUDA runtime available to torch.")
    if not torch.version.cuda:
        print(
            "  Note: this torch is a CPU-only build "
            f"({torch.__version__}). A CUDA build is needed to use a local "
            "GPU; otherwise train on Kaggle/Colab (ROADMAP §4)."
        )
    print("  CPU training at 640px will take many hours per run.")
    return "cpu"


def _check_data(data_yaml: Path):
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset config not found: {data_yaml}\n"
            "Build the dataset first — see ROADMAP §3:\n"
            "  python scripts/idd_to_yolo.py --idd-root <path> --out-root data/detection\n"
            "  python scripts/exdark_to_yolo.py convert --exdark-root <path>\n"
            "  python scripts/exdark_to_yolo.py darken"
        )


def train_detection(args):
    """Two-stage fine-tune: frozen-backbone warm-up, then full unfreeze."""
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    _check_data(data_yaml)
    device = _resolve_device(args.device)

    warmup = min(args.warmup_epochs, args.epochs)
    remaining = args.epochs - warmup

    common = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        exist_ok=True,
    )

    # Stage 1 — head only. `freeze=10` freezes the first 10 modules, which
    # is the backbone for the v8 nano architecture.
    print(f"\n=== Stage 1/2: {warmup} epochs, backbone frozen ===")
    model = YOLO(args.weights)
    model.train(name=f"{args.name}_warmup", epochs=warmup, freeze=10, **common)

    if remaining <= 0:
        print("\nNo unfrozen epochs requested — stopping after warm-up.")
        return

    # Stage 2 — everything trainable, starting from the warm-up's best.
    warmup_best = Path(args.project) / f"{args.name}_warmup" / "weights" / "best.pt"
    if not warmup_best.exists():
        raise SystemExit(f"Warm-up produced no weights at {warmup_best}")

    print(f"\n=== Stage 2/2: {remaining} epochs, all layers trainable ===")
    model = YOLO(str(warmup_best))
    results = model.train(name=args.name, epochs=remaining, freeze=None, **common)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Evaluate day vs night separately before trusting the headline number:")
    print(f"  python scripts/evaluate.py detection --weights {best} --data {data_yaml}")
    return results


def train_plate(args):
    """Single-class plate localiser — one stage, no freeze schedule."""
    from ultralytics import YOLO

    data_yaml = Path(args.data)
    _check_data(data_yaml)
    device = _resolve_device(args.device)

    print(f"\n=== Training plate localiser: {args.epochs} epochs ===")
    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Compare train vs val mAP in the run summary — a large gap on a "
          "single-class dataset this size means overfitting, not success.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IBVAP's YOLOv8 models")
    sub = parser.add_subparsers(dest="model", required=True)

    def add_common(p, default_data, default_name, default_epochs):
        p.add_argument("--data", type=str, default=default_data)
        p.add_argument("--weights", type=str, default="yolov8n.pt",
                       help="Starting weights (keep nano for edge deployment)")
        p.add_argument("--epochs", type=int, default=default_epochs)
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--batch", type=int, default=16,
                       help="16 fits a free-tier T4; drop to 8 on a 4GB laptop GPU")
        p.add_argument("--device", type=str, default="auto",
                       help="'auto', 'cpu', or a CUDA index like '0'")
        p.add_argument("--patience", type=int, default=15,
                       help="Early-stopping patience on val mAP")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--project", type=str, default="runs/detect")
        p.add_argument("--name", type=str, default=default_name)

    p_det = sub.add_parser("detection", help="Person/vehicle detector (IDD + ExDark)")
    add_common(p_det, "data/detection/data.yaml", "ibvap_detection", 80)
    p_det.add_argument("--warmup-epochs", type=int, default=10,
                       help="Epochs to train with the backbone frozen")

    p_plate = sub.add_parser("plate", help="Single-class plate localiser")
    add_common(p_plate, "data/anpr/data.yaml", "ibvap_plate", 50)

    args = parser.parse_args()
    if args.model == "detection":
        train_detection(args)
    else:
        train_plate(args)


if __name__ == "__main__":
    main()
