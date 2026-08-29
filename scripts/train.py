"""
IBVAP Training Pipeline
Train models using the provided datasets
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TrainingPipeline:
    """
    Training pipeline for IBVAP models
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Dataset paths
        self.data_dir = Path("data")
        
    def _load_config(self) -> Dict:
        """Load training configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        else:
            print(f"Config not found at {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            "models": {
                "detection": {
                    "name": "yolov8n",
                    "epochs": 100,
                    "batch_size": 16,
                    "img_size": 640,
                    "learning_rate": 0.001
                },
                "anpr": {
                    "name": "paddleocr",
                    "epochs": 50,
                    "batch_size": 32
                },
                "surveillance": {
                    "name": "slowfast",
                    "epochs": 100,
                    "batch_size": 8
                }
            },
            "datasets": {
                "detection": ["idd", "bdd100k"],
                "anpr": ["indian_number_plates", "ufpr_alpr"],
                "surveillance": ["virat", "ucf_crime"]
            }
        }
    
    def train_detection_model(self, dataset_name: str = "idd"):
        """Train object detection model using YOLOv8"""
        print("\n" + "=" * 60)
        print(f"Training Detection Model on {dataset_name}")
        print("=" * 60)
        
        # Check if dataset exists
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset not found at {dataset_dir}")
            print("Please run: python scripts/prepare_datasets.py --prepare")
            return False
        
        # Check for YOLO format data
        yolo_dir = dataset_dir / "yolo"
        if not yolo_dir.exists():
            print("YOLO format data not found. Converting...")
            from scripts.prepare_datasets import DatasetPreparer
            preparer = DatasetPreparer(str(self.data_dir))
            preparer.convert_to_yolo_format(dataset_name)
        
        # Create YOLO dataset config
        dataset_yaml = yolo_dir / "dataset.yaml"
        self._create_yolo_dataset_config(dataset_name, dataset_yaml)
        
        # Training command
        model_config = self.config["models"]["detection"]
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {model_config['name']}")
        print(f"  Epochs: {model_config['epochs']}")
        print(f"  Batch Size: {model_config['batch_size']}")
        print(f"  Image Size: {model_config['img_size']}")
        print(f"  Learning Rate: {model_config['learning_rate']}")
        
        # Generate training script
        train_script = f"""
# YOLOv8 Training Script for {dataset_name}
from ultralytics import YOLO

# Load model
model = YOLO('{model_config['name']}.pt')

# Train
results = model.train(
    data='{dataset_yaml}',
    epochs={model_config['epochs']},
    batch={model_config['batch_size']},
    imgsz={model_config['img_size']},
    lr0={model_config['learning_rate']},
    project='outputs/detection',
    name='{dataset_name}_{model_config['name']}',
    exist_ok=True
)

print("Training complete!")
print(f"Results saved to: outputs/detection/{dataset_name}_{model_config['name']}")
"""
        
        script_path = Path(f"scripts/run_train_detection_{dataset_name}.py")
        with open(script_path, "w") as f:
            f.write(train_script)
        
        print(f"\nTraining script created: {script_path}")
        print(f"Run: python {script_path}")
        
        return True
    
    def _create_yolo_dataset_config(self, dataset_name: str, output_path: Path):
        """Create YOLO dataset configuration file"""
        dataset_dir = self.data_dir / dataset_name
        
        # Load dataset info
        info_path = dataset_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            classes = info.get("classes", [])
        else:
            classes = ["object"]  # Default class
        
        config = {
            "path": str(dataset_dir / "yolo"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": classes
        }
        
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created YOLO config: {output_path}")
    
    def train_anpr_model(self, dataset_name: str = "indian_number_plates"):
        """Train ANPR model using PaddleOCR"""
        print("\n" + "=" * 60)
        print(f"Training ANPR Model on {dataset_name}")
        print("=" * 60)
        
        # Check if dataset exists
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset not found at {dataset_dir}")
            return False
        
        # Generate training script
        model_config = self.config["models"]["anpr"]
        
        train_script = f"""
# PaddleOCR Training Script for ANPR
# Note: PaddleOCR requires PaddlePaddle framework

print("ANPR Training Pipeline")
print("=" * 40)
print(f"Dataset: {dataset_name}")
print(f"Epochs: {model_config['epochs']}")
print(f"Batch Size: {model_config['batch_size']}")
print()
print("To train PaddleOCR:")
print("1. Install PaddlePaddle: pip install paddlepaddle")
print("2. Install PaddleOCR: pip install paddleocr")
print("3. Follow: https://github.com/PaddlePaddle/PaddleOCR/doc/doc_en/training_en.md")
print()
print("For this demo, we'll use pre-trained PaddleOCR with fine-tuning.")
"""
        
        script_path = Path(f"scripts/run_train_anpr_{dataset_name}.py")
        with open(script_path, "w") as f:
            f.write(train_script)
        
        print(f"\nTraining script created: {script_path}")
        print(f"Run: python {script_path}")
        
        return True
    
    def train_surveillance_model(self, dataset_name: str = "virat"):
        """Train surveillance activity recognition model"""
        print("\n" + "=" * 60)
        print(f"Training Surveillance Model on {dataset_name}")
        print("=" * 60)
        
        # Check if dataset exists
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset not found at {dataset_dir}")
            return False
        
        model_config = self.config["models"]["surveillance"]
        
        train_script = f"""
# SlowFast Training Script for Surveillance
# Activity Recognition using SlowFast Network

print("Surveillance Activity Recognition Training")
print("=" * 40)
print(f"Dataset: {dataset_name}")
print(f"Model: {model_config['name']}")
print(f"Epochs: {model_config['epochs']}")
print(f"Batch Size: {model_config['batch_size']}")
print()
print("To train SlowFast:")
print("1. Install PyTorch: pip install torch torchvision")
print("2. Install PySlowFast: git clone https://github.com/facebookresearch/SlowFast")
print("3. Follow: https://github.com/facebookresearch/SlowFast/blob/main/GETTING_STARTED.md")
"""
        
        script_path = Path(f"scripts/run_train_surveillance_{dataset_name}.py")
        with open(script_path, "w") as f:
            f.write(train_script)
        
        print(f"\nTraining script created: {script_path}")
        print(f"Run: python {script_path}")
        
        return True
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "=" * 60)
        print("IBVAP Full Training Pipeline")
        print("=" * 60)
        
        # Train detection model
        detection_datasets = self.config["datasets"]["detection"]
        for dataset in detection_datasets:
            self.train_detection_model(dataset)
        
        # Train ANPR model
        anpr_datasets = self.config["datasets"]["anpr"]
        for dataset in anpr_datasets:
            self.train_anpr_model(dataset)
        
        # Train surveillance model
        surveillance_datasets = self.config["datasets"]["surveillance"]
        for dataset in surveillance_datasets:
            self.train_surveillance_model(dataset)
        
        print("\n" + "=" * 60)
        print("All training scripts generated!")
        print("=" * 60)
    
    def print_training_guide(self):
        """Print complete training guide"""
        guide = """
╔══════════════════════════════════════════════════════════════╗
║              IBVAP TRAINING GUIDE                           ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Download Datasets
═══════════════════════════════════════════════════════════════
  python scripts/prepare_datasets.py --download-scripts
  bash data/download_scripts/download_all.sh

STEP 2: Prepare Datasets
═══════════════════════════════════════════════════════════════
  python scripts/prepare_datasets.py --prepare

STEP 3: Train Models
═══════════════════════════════════════════════════════════════

  A) Detection Model (YOLOv8)
  ─────────────────────────────
  python scripts/train.py --model detection --dataset idd
  
  Or train on all detection datasets:
  python scripts/train.py --model detection --all

  B) ANPR Model (PaddleOCR)
  ─────────────────────────────
  python scripts/train.py --model anpr --dataset indian_number_plates
  
  Or train on all ANPR datasets:
  python scripts/train.py --model anpr --all

  C) Surveillance Model (SlowFast)
  ─────────────────────────────
  python scripts/train.py --model surveillance --dataset virat
  
  Or train on all surveillance datasets:
  python scripts/train.py --model surveillance --all

STEP 4: Evaluate Models
═══════════════════════════════════════════════════════════════
  python scripts/evaluate.py --model detection
  python scripts/evaluate.py --model anpr
  python scripts/evaluate.py --model surveillance

STEP 5: Export Models
═══════════════════════════════════════════════════════════════
  python scripts/export.py --model detection --format onnx
  python scripts/export.py --model anpr --format onnx

╔══════════════════════════════════════════════════════════════╗
║                    QUICK START                               ║
╚══════════════════════════════════════════════════════════════╝

For a quick demo without full training:

1. Use pre-trained YOLOv8:
   python app.py --demo

2. The demo uses simulated detections.
   Real training requires downloading datasets first.

╔══════════════════════════════════════════════════════════════╗
║                 DATASET REQUIREMENTS                         ║
╚══════════════════════════════════════════════════════════════╝

Required for full training:
• IDD Dataset (~2.5 GB)
• Indian Number Plates (~3 GB)
• VIRAT Dataset (~50 GB, requires registration)
• ExDark Dataset (~0.8 GB)
• WIDER FACE (~3.5 GB)

Optional for enhanced training:
• BDD100K (~100 GB)
• UCF-Crime (~15 GB)
• MEVA (~470 GB)
"""
        print(guide)


def main():
    parser = argparse.ArgumentParser(description="IBVAP Training Pipeline")
    parser.add_argument("--model", choices=["detection", "anpr", "surveillance", "all"],
                       help="Model type to train")
    parser.add_argument("--dataset", help="Specific dataset to use")
    parser.add_argument("--all", action="store_true", help="Train on all datasets")
    parser.add_argument("--guide", action="store_true", help="Print training guide")
    parser.add_argument("--config", default="config/training_config.yaml",
                       help="Training config file")
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(args.config)
    
    if args.guide:
        pipeline.print_training_guide()
    elif args.model == "all" or args.all:
        pipeline.train_all_models()
    elif args.model == "detection":
        pipeline.train_detection_model(args.dataset or "idd")
    elif args.model == "anpr":
        pipeline.train_anpr_model(args.dataset or "indian_number_plates")
    elif args.model == "surveillance":
        pipeline.train_surveillance_model(args.dataset or "virat")
    else:
        pipeline.print_training_guide()


if __name__ == "__main__":
    main()
