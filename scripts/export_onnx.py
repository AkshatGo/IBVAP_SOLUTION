"""
IBVAP — export fine-tuned weights to ONNX (ROADMAP §5.1).

Produces the portable artifact the Jetson/TensorRT roadmap depends on.
Exporting is not the same as having benchmarked on Jetson hardware — this
script gets you a real .onnx file you can point at, and nothing more. Say
it that way in the deck.

Usage:
    python scripts/export_onnx.py --weights runs/detect/ibvap_detection/weights/best.pt
    python scripts/export_onnx.py --weights runs/detect/ibvap_plate/weights/best.pt \
        --imgsz 640 --opset 12 --simplify

Verification runs by default: the exported graph is loaded back through
onnxruntime (if installed) and given one dummy frame, so a broken export
fails here rather than on the edge device.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


def export(weights: Path, imgsz: int, opset: int, simplify: bool, half: bool,
           dynamic: bool, out_dir: Path, name: str = None):
    from ultralytics import YOLO

    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train a model first: python scripts/train.py detection"
        )

    print(f"Exporting {weights} -> ONNX (imgsz={imgsz}, opset={opset})")
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=dynamic,
    )

    exported_path = Path(exported)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every Ultralytics run produces "best.pt", so a naive move drops each
    # export on top of the last one. Name the artifact after the run
    # directory (runs/ibvap_plate/weights/best.pt -> ibvap_plate.onnx)
    # unless the caller asked for something specific.
    if name:
        stem = name
    elif exported_path.stem in {"best", "last"}:
        stem = weights.parent.parent.name
    else:
        stem = exported_path.stem
    destination = out_dir / f"{stem}{exported_path.suffix}"
    if exported_path.resolve() != destination.resolve():
        shutil.move(str(exported_path), destination)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Wrote {destination} ({size_mb:.1f} MB)")
    return destination


def verify(onnx_path: Path, imgsz: int):
    """Load the exported graph and run one dummy frame through it."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSkipping verification — onnxruntime is not installed.")
        print("  pip install onnxruntime   # then re-run with --verify")
        return None

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

    outputs = session.run(None, {input_meta.name: dummy})
    shapes = [tuple(o.shape) for o in outputs]
    print(f"\nVerified: input '{input_meta.name}' {input_meta.shape} -> outputs {shapes}")
    return shapes


def main():
    parser = argparse.ArgumentParser(description="Export IBVAP weights to ONNX")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to a .pt checkpoint (e.g. .../weights/best.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12,
                        help="12 is widely supported by TensorRT builds")
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier on the graph (needs onnxsim)")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 — GPU inference only, not CPU")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic batch/size axes; leave off for TensorRT")
    parser.add_argument("--out-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--name", type=str, default=None,
                        help="Output filename stem; defaults to the run "
                             "directory name so exports do not collide")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="Skip the onnxruntime load-and-run check")
    args = parser.parse_args()

    onnx_path = export(args.weights, args.imgsz, args.opset, args.simplify,
                       args.half, args.dynamic, args.out_dir, args.name)
    if args.verify:
        verify(onnx_path, args.imgsz)

    print("\nNext: point the edge pipeline at the checkpoint you exported —")
    print("  IBVAP_DETECTION_MODEL=<weights.pt> python main.py demo")


if __name__ == "__main__":
    main()
