"""
IBVAP Dataset Preparation Script
Download, prepare, and organize all training datasets
"""

import os
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class DatasetPreparer:
    """
    Prepare all IBVAP datasets for training
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            # ============================================================
            # 1. Vehicle Detection Datasets
            # ============================================================
            "idd": {
                "name": "Indian Driving Dataset",
                "category": "detection",
                "subdir": "idd",
                "structure": {
                    "images": ["images/train", "images/val", "images/test"],
                    "annotations": ["annotations"]
                },
                "classes": [
                    "auto_rickshaw", "bus", "car", "motorcycle", "truck", "bicycle",
                    "person", "traffic_light", "traffic_sign", "road_divider",
                    "lane_divider", "parking", "rider", "trailer", "vehicle_group",
                    "two_wheeler", "three_wheeler", "construction_vehicle"
                ],
                "download_instructions": """
                1. Visit: https://iith.ac.in/projects/idddataset/
                2. Register and download the dataset
                3. Extract to: data/idd/
                4. Or use Kaggle: kaggle datasets download -d insaan/indian-driving-dataset -p data/idd/
                """
            },
            
            "bdd100k": {
                "name": "BDD100K",
                "category": "detection",
                "subdir": "bdd100k",
                "structure": {
                    "images": ["images/train", "images/val", "images/test"],
                    "annotations": ["labels"]
                },
                "classes": [
                    "pedestrian", "rider", "car", "truck", "bus", "train",
                    "motorcycle", "bicycle", "traffic_light", "traffic_sign"
                ],
                "download_instructions": """
                1. Visit: https://bdd-data.berkeley.edu/
                2. Register and download
                3. Extract to: data/bdd100k/
                """
            },
            
            # ============================================================
            # 2. ANPR Datasets
            # ============================================================
            "indian_number_plates": {
                "name": "Indian Number Plates Dataset",
                "category": "anpr",
                "subdir": "indian_number_plates",
                "structure": {
                    "images": ["images"],
                    "annotations": ["annotations"]
                },
                "classes": ["license_plate", "vehicle"],
                "download_instructions": """
                1. Visit: https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset
                2. Download using Kaggle CLI:
                   kaggle datasets download -d dataclusterlabs/indian-number-plates-dataset -p data/indian_number_plates/
                3. Extract the downloaded zip
                """
            },
            
            "ufpr_alpr": {
                "name": "UFPR-ALPR Dataset",
                "category": "anpr",
                "subdir": "ufpr_alpr",
                "structure": {
                    "images": ["images"],
                    "annotations": ["annotations"]
                },
                "classes": ["license_plate", "vehicle"],
                "download_instructions": """
                1. Visit: https://www.kaggle.com/datasets/andrewmvd/ufpr-alpr
                2. Download using Kaggle CLI:
                   kaggle datasets download -d andrewmvd/ufpr-alpr -p data/ufpr_alpr/
                3. Extract the downloaded zip
                """
            },
            
            # ============================================================
            # 3. Surveillance Datasets
            # ============================================================
            "virat": {
                "name": "VIRAT Video Surveillance Dataset",
                "category": "surveillance",
                "subdir": "virat",
                "structure": {
                    "videos": ["videos"],
                    "annotations": ["annotations"]
                },
                "classes": [
                    "person", "vehicle", "activity",
                    "intrusion", "loitering", "suspicious"
                ],
                "download_instructions": """
                1. Visit: https://viratdata.org/
                2. Register for access
                3. Download and extract to: data/virat/
                """
            },
            
            "ucf_crime": {
                "name": "UCF-Crime Dataset",
                "category": "anomaly",
                "subdir": "ucf_crime",
                "structure": {
                    "videos": ["Normal", "Abuse", "Arrest", "Arson", "Assault",
                              "Burglary", "Explosion", "Fighting", "Road Accidents",
                              "Robbery", "Shooting", "Stealing", "Vandalism"],
                    "annotations": ["annotations"]
                },
                "classes": [
                    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
                    "fighting", "normal", "road_accidents", "robbery", "shooting",
                    "stealing", "vandalism"
                ],
                "download_instructions": """
                1. Visit: https://www.crcv.ucf.edu/data/UCF-Crime.php
                2. Download the dataset
                3. Extract to: data/ucf_crime/
                """
            },
            
            "iitm_hetra": {
                "name": "IITM-HeTra Video Surveillance Dataset",
                "category": "surveillance",
                "subdir": "iitm_hetra",
                "structure": {
                    "videos": ["videos"],
                    "annotations": ["annotations"]
                },
                "classes": ["vehicle", "person", "pedestrian"],
                "download_instructions": """
                1. Visit: https://github.com/suriya-iitm/HeTra
                2. Clone the repository
                3. Download dataset and extract to: data/iitm_hetra/
                """
            },
            
            # ============================================================
            # 4. Low-Light Dataset
            # ============================================================
            "exdark": {
                "name": "ExDark (Exclusively Dark Image Dataset)",
                "category": "low_light",
                "subdir": "exdark",
                "structure": {
                    "images": ["images"],
                    "annotations": ["annotations"]
                },
                "classes": [
                    "bicycle", "boat", "bottle", "bus", "car", "cat", "chair",
                    "cup", "dog", "motorbike", "people", "table"
                ],
                "download_instructions": """
                1. Visit: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
                2. Clone the repository or download ZIP
                3. Extract to: data/exdark/
                """
            },
            
            # ============================================================
            # 5. Face Detection Dataset
            # ============================================================
            "wider_face": {
                "name": "WIDER FACE Dataset",
                "category": "face_detection",
                "subdir": "wider_face",
                "structure": {
                    "images": ["WIDER_train/images", "WIDER_val/images", "WIDER_test/images"],
                    "annotations": ["wider_face_split"]
                },
                "classes": ["face"],
                "download_instructions": """
                1. Visit: https://shuoyang1213.me/WIDERFACE/
                2. Download images and annotations
                3. Extract to: data/wider_face/
                """
            },
            
            # ============================================================
            # 6. Tracking Datasets
            # ============================================================
            "idd_temporal": {
                "name": "IDD Temporal",
                "category": "tracking",
                "subdir": "idd_temporal",
                "structure": {
                    "sequences": ["sequences"],
                    "annotations": ["annotations"]
                },
                "classes": ["vehicle", "person"],
                "download_instructions": """
                1. Visit: https://idd.insaan.co.in/
                2. Download temporal sequences
                3. Extract to: data/idd_temporal/
                """
            },
            
            # ============================================================
            # 7. Custom Border Dataset (to be collected)
            # ============================================================
            "custom_border": {
                "name": "Custom Border Surveillance Dataset",
                "category": "custom",
                "subdir": "custom_border",
                "structure": {
                    "videos": ["videos"],
                    "images": ["images"],
                    "annotations": ["annotations"]
                },
                "classes": [
                    "boundary_crossing", "perimeter_intrusion", "loitering",
                    "suspicious_movement", "vehicle_intrusion", "night_activity",
                    "person", "vehicle", "animal", "patrol"
                ],
                "download_instructions": """
                This dataset needs to be collected and annotated:
                1. Record border surveillance scenarios
                2. Annotate using CVAT or LabelImg
                3. Place in: data/custom_border/
                """
            },
        }
    
    def create_directory_structure(self, dataset_name: str) -> bool:
        """Create directory structure for a dataset"""
        if dataset_name not in self.datasets:
            print(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_config["subdir"]
        
        print(f"\nCreating directory structure for {dataset_config['name']}...")
        
        # Create main directories
        for category, paths in dataset_config["structure"].items():
            for path in paths:
                full_path = dataset_dir / path
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {full_path}")
        
        # Create dataset info file
        info = {
            "name": dataset_config["name"],
            "category": dataset_config["category"],
            "subdir": dataset_config["subdir"],
            "classes": dataset_config["classes"],
            "structure": dataset_config["structure"],
            "prepared": True
        }
        
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"  Created: {dataset_dir}/dataset_info.json")
        
        return True
    
    def create_split_file(self, dataset_name: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Create train/val/test split file"""
        if dataset_name not in self.datasets:
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_config["subdir"]
        
        # Find all image/video files
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi'}
        files = []
        
        for ext in extensions:
            files.extend(dataset_dir.rglob(f"*{ext}"))
        
        if not files:
            print(f"No files found in {dataset_dir}")
            return False
        
        # Shuffle and split
        import random
        random.shuffle(files)
        
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        splits = {
            "train": [str(f.relative_to(dataset_dir)) for f in files[:train_end]],
            "val": [str(f.relative_to(dataset_dir)) for f in files[train_end:val_end]],
            "test": [str(f.relative_to(dataset_dir)) for f in files[val_end:]]
        }
        
        # Save splits
        with open(dataset_dir / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)
        
        print(f"\nCreated splits for {dataset_name}:")
        print(f"  Train: {len(splits['train'])} files")
        print(f"  Val: {len(splits['val'])} files")
        print(f"  Test: {len(splits['test'])} files")
        
        return True
    
    def convert_to_yolo_format(self, dataset_name: str):
        """Convert annotations to YOLO format"""
        if dataset_name not in self.datasets:
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_config["subdir"]
        classes = dataset_config["classes"]
        
        print(f"\nConverting {dataset_name} to YOLO format...")
        
        # Create YOLO directories
        yolo_dir = dataset_dir / "yolo"
        for split in ["train", "val", "test"]:
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Load splits if they exist
        splits_file = dataset_dir / "splits.json"
        if splits_file.exists():
            with open(splits_file) as f:
                splits = json.load(f)
        else:
            print("  No splits.json found. Run create_split_file first.")
            return False
        
        # Process each split
        for split_name, files in splits.items():
            print(f"  Processing {split_name} split ({len(files)} files)...")
            
            for file_path in files:
                src_path = dataset_dir / file_path
                
                if not src_path.exists():
                    continue
                
                # Copy image
                dst_img = yolo_dir / "images" / split_name / src_path.name
                shutil.copy2(src_path, dst_img)
                
                # Create placeholder label (to be filled with actual annotations)
                label_path = yolo_dir / "labels" / split_name / (src_path.stem + ".txt")
                if not label_path.exists():
                    with open(label_path, "w") as f:
                        pass  # Empty label file
        
        print(f"\nYOLO format created at: {yolo_dir}")
        return True
    
    def generate_download_script(self, dataset_name: str) -> str:
        """Generate download script for a dataset"""
        if dataset_name not in self.datasets:
            return ""
        
        dataset_config = self.datasets[dataset_name]
        
        script = f"""
# Download Script for {dataset_config['name']}
# Category: {dataset_config['category']}

echo "Downloading {dataset_config['name']}..."

{dataset_config['download_instructions']}

echo "Done! Extract to: data/{dataset_config['subdir']}/"
"""
        return script
    
    def generate_all_download_scripts(self):
        """Generate download scripts for all datasets"""
        scripts_dir = self.data_dir / "download_scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        for dataset_name in self.datasets:
            script = self.generate_download_script(dataset_name)
            script_path = scripts_dir / f"download_{dataset_name}.sh"
            
            with open(script_path, "w") as f:
                f.write(script)
            
            os.chmod(script_path, 0o755)
        
        # Create master download script
        master_script = "#!/bin/bash\n\n# IBVAP Dataset Download Master Script\n\n"
        
        for dataset_name in self.datasets:
            master_script += f'echo "Download {self.datasets[dataset_name]["name"]}?"\n'
            master_script += f'read -p "Press Enter to continue, Ctrl+C to skip..."\n'
            master_script += f'bash {scripts_dir}/download_{dataset_name}.sh\n\n'
        
        master_path = scripts_dir / "download_all.sh"
        with open(master_path, "w") as f:
            f.write(master_script)
        
        os.chmod(master_path, 0o755)
        
        print(f"\nDownload scripts created at: {scripts_dir}")
        print(f"Run: bash {master_path}")
    
    def prepare_all_datasets(self):
        """Prepare all datasets"""
        print("=" * 60)
        print("IBVAP Dataset Preparation")
        print("=" * 60)
        
        for dataset_name in self.datasets:
            print(f"\n{'-'*60}")
            print(f"Preparing: {self.datasets[dataset_name]['name']}")
            print(f"{'-'*60}")
            
            # Create directory structure
            self.create_directory_structure(dataset_name)
            
            # Create splits
            self.create_split_file(dataset_name)
            
            # Convert to YOLO format (for detection datasets)
            if self.datasets[dataset_name]["category"] in ["detection", "low_light"]:
                self.convert_to_yolo_format(dataset_name)
        
        print("\n" + "=" * 60)
        print("Dataset preparation complete!")
        print("=" * 60)
        
        # Generate download scripts
        self.generate_all_download_scripts()
    
    def print_summary(self):
        """Print summary of all datasets"""
        print("\n" + "=" * 60)
        print("IBVAP DATASET CATALOG")
        print("=" * 60)
        
        total_classes = 0
        
        for category in ["detection", "anpr", "surveillance", "anomaly", "low_light", "face_detection", "tracking", "custom"]:
            datasets = [(k, v) for k, v in self.datasets.items() if v["category"] == category]
            
            if datasets:
                print(f"\n{category.upper().replace('_', ' ')}:")
                print("-" * 40)
                
                for name, config in datasets:
                    print(f"  {name:20} | {len(config['classes']):2} classes | {config['name']}")
                    total_classes += len(config['classes'])
        
        print("\n" + "=" * 60)
        print(f"Total datasets: {len(self.datasets)}")
        print(f"Total classes: {total_classes}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Preparation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--prepare", action="store_true", help="Prepare all datasets")
    parser.add_argument("--dataset", help="Prepare specific dataset")
    parser.add_argument("--splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--yolo", action="store_true", help="Convert to YOLO format")
    parser.add_argument("--download-scripts", action="store_true", help="Generate download scripts")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.data_dir)
    
    if args.summary:
        preparer.print_summary()
    elif args.download_scripts:
        preparer.generate_all_download_scripts()
    elif args.dataset:
        preparer.create_directory_structure(args.dataset)
        if args.splits:
            preparer.create_split_file(args.dataset)
        if args.yolo:
            preparer.convert_to_yolo_format(args.dataset)
    elif args.prepare:
        preparer.prepare_all_datasets()
    else:
        preparer.print_summary()
        preparer.generate_all_download_scripts()


if __name__ == "__main__":
    main()
