"""
Convert Indian Number Plates Dataset to YOLO Format and Train
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
import yaml

# ============================================================
# Step 1: Parse XML annotations and convert to YOLO format
# ============================================================

def parse_xml_annotation(xml_path):
    """Parse XML annotation and return bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    # Remove extension if present
    filename = Path(filename).stem
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'number_plate':
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Class 0 = number_plate
            bboxes.append((0, x_center, y_center, width, height))
    
    return filename, bboxes, img_width, img_height


def convert_dataset():
    """Convert XML annotations to YOLO format"""
    print("=" * 60)
    print("CONVERTING INDIAN NUMBER PLATES TO YOLO FORMAT")
    print("=" * 60)
    
    data_dir = Path("data/indian_number_plates")
    xml_dir = data_dir / "Annotations/Annotations"
    img_dir = data_dir / "Indian_Number_Plates/Sample_Images"
    
    # Create YOLO directory structure
    yolo_dir = data_dir / "yolo"
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    
    # Parse all annotations
    data = []
    for xml_file in xml_files:
        try:
            filename, bboxes, w, h = parse_xml_annotation(xml_file)
            if bboxes:
                # Find corresponding image
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = img_dir / (filename + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path:
                    data.append({
                        'img_path': img_path,
                        'bboxes': bboxes,
                        'filename': filename
                    })
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    
    print(f"Parsed {len(data)} images with annotations")
    
    # Shuffle and split
    random.shuffle(data)
    train_end = int(len(data) * 0.8)
    val_end = int(len(data) * 0.9)
    
    splits = {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }
    
    # Process each split
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name}: {len(split_data)} images")
        
        for item in split_data:
            # Copy image
            dst_img = yolo_dir / "images" / split_name / item['img_path'].name
            shutil.copy2(item['img_path'], dst_img)
            
            # Create label file
            label_path = yolo_dir / "labels" / split_name / (item['img_path'].stem + ".txt")
            with open(label_path, 'w') as f:
                for bbox in item['bboxes']:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(yolo_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['number_plate']
    }
    
    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"\nDataset YAML created: {yaml_path}")
    print(f"\nSplit sizes:")
    for split, data in splits.items():
        print(f"  {split}: {len(data)} images")
    
    return yaml_path


# ============================================================
# Step 2: Train YOLOv8
# ============================================================

def train_model(dataset_yaml):
    """Train YOLOv8 on the converted dataset"""
    print("\n" + "=" * 60)
    print("TRAINING YOLOv8 ON INDIAN NUMBER PLATES")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Load YOLOv8 nano model (smallest, fastest)
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        name='anpr_indian_plates',
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/anpr_indian_plates/")
    
    return results


# ============================================================
# Step 3: Validate
# ============================================================

def validate_model(dataset_yaml):
    """Validate the trained model"""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Load best model
    model = YOLO('runs/detect/anpr_indian_plates/weights/best.pt')
    
    # Validate
    results = model.val(data=str(dataset_yaml))
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="Convert dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.all or args.convert:
        dataset_yaml = convert_dataset()
    
    if args.all or args.train:
        if 'dataset_yaml' not in locals():
            dataset_yaml = Path("data/indian_number_plates/yolo/dataset.yaml")
        train_model(dataset_yaml)
    
    if args.all or args.validate:
        if 'dataset_yaml' not in locals():
            dataset_yaml = Path("data/indian_number_plates/yolo/dataset.yaml")
        validate_model(dataset_yaml)
