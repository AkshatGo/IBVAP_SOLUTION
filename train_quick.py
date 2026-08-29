"""
Quick YOLOv8 Training Script
"""

import sys
import os
from pathlib import Path

# Set up paths
os.chdir(Path(__file__).parent)

print("=" * 60)
print("IBVAP - YOLOv8 Training")
print("=" * 60)

try:
    # Test imports
    print("\n[1/4] Testing imports...")
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    
    from ultralytics import YOLO
    print(f"  ✓ Ultralytics imported")
    
    # Check dataset
    print("\n[2/4] Checking dataset...")
    dataset_yaml = Path("data/indian_number_plates/yolo/dataset.yaml")
    if not dataset_yaml.exists():
        print("  ✗ Dataset not found. Run conversion first.")
        sys.exit(1)
    
    import yaml
    with open(dataset_yaml) as f:
        config = yaml.safe_load(f)
    
    train_images = Path(config['path']) / config['train']
    num_train = len(list(train_images.glob('*')))
    print(f"  ✓ Dataset found: {num_train} training images")
    
    # Load model
    print("\n[3/4] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("  ✓ Model loaded")
    
    # Train
    print("\n[4/4] Starting training (25 epochs)...")
    print("  This may take a while on CPU...")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=25,
        imgsz=640,
        batch=4,  # Small batch for CPU
        name='ibvap_anpr',
        exist_ok=True,
        patience=10,
        verbose=True,
        device='cpu'
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: runs/detect/ibvap_anpr/")
    
    # Check for best model
    best_model = Path("runs/detect/ibvap_anpr/weights/best.pt")
    if best_model.exists():
        print(f"Best model: {best_model}")
        
        # Validate
        print("\nRunning validation...")
        model = YOLO(str(best_model))
        val_results = model.val(data=str(dataset_yaml))
        
        print(f"\nValidation Results:")
        print(f"  mAP50: {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
