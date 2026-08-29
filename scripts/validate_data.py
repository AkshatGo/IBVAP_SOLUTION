"""
IBVAP Data Validation Utilities
Validate dataset integrity and quality
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ValidationResult:
    """Result of data validation"""
    dataset_name: str
    total_samples: int
    valid_samples: int
    invalid_samples: int
    errors: List[str]
    warnings: List[str]
    statistics: Dict
    
    @property
    def is_valid(self) -> bool:
        return self.invalid_samples == 0
    
    @property
    def success_rate(self) -> float:
        return self.valid_samples / max(self.total_samples, 1) * 100


class DataValidator:
    """
    Validate dataset integrity and quality
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """Validate a single image file"""
        try:
            if not os.path.exists(image_path):
                return False, f"File not found: {image_path}"
            
            # Try to read with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False, f"Cannot read image: {image_path}"
            
            # Check dimensions
            if img.shape[0] < 10 or img.shape[1] < 10:
                return False, f"Image too small: {image_path} ({img.shape})"
            
            if img.shape[0] > 10000 or img.shape[1] > 10000:
                return False, f"Image too large: {image_path} ({img.shape})"
            
            # Check if all zeros (corrupted)
            if np.sum(img) == 0:
                return False, f"Image is all zeros: {image_path}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Error reading {image_path}: {str(e)}"
    
    def validate_bbox(self, bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """Validate a bounding box"""
        try:
            if len(bbox) != 4:
                return False, f"Invalid bbox length: {len(bbox)}"
            
            x1, y1, x2, y2 = bbox
            
            # Check coordinates
            if x1 >= x2 or y1 >= y2:
                return False, f"Invalid bbox coordinates: {bbox}"
            
            if x1 < 0 or y1 < 0:
                return False, f"Negative coordinates: {bbox}"
            
            if x2 > img_shape[1] or y2 > img_shape[0]:
                return False, f"Bbox outside image: {bbox}"
            
            # Check minimum size
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                return False, f"Bbox too small: {bbox}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Error validating bbox: {str(e)}"
    
    def validate_annotation_file(self, annotation_path: str) -> ValidationResult:
        """Validate an annotation file"""
        errors = []
        warnings = []
        
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return ValidationResult(
                dataset_name=annotation_path,
                total_samples=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Cannot parse annotation file: {str(e)}"],
                warnings=[],
                statistics={}
            )
        
        # Handle different annotation formats
        if isinstance(data, dict):
            # COCO format
            images = data.get('images', [])
            annotations = data.get('annotations', [])
            
            total = len(images)
            valid = 0
            invalid = 0
            
            # Check image IDs
            image_ids = {img['id'] for img in images}
            
            for ann in annotations:
                if ann.get('image_id') not in image_ids:
                    warnings.append(f"Annotation references non-existent image: {ann.get('image_id')}")
                
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    if len(bbox) != 4:
                        errors.append(f"Invalid bbox in annotation: {ann['id']}")
        
        elif isinstance(data, list):
            # Custom format
            total = len(data)
            valid = sum(1 for item in data if 'image' in item)
            invalid = total - valid
        
        else:
            total = 0
            valid = 0
            invalid = 0
            errors.append("Unknown annotation format")
        
        return ValidationResult(
            dataset_name=annotation_path,
            total_samples=total,
            valid_samples=valid,
            invalid_samples=invalid,
            errors=errors,
            warnings=warnings,
            statistics={'format': type(data).__name__}
        )
    
    def validate_dataset(self, dataset_name: str) -> ValidationResult:
        """Validate an entire dataset"""
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            return ValidationResult(
                dataset_name=dataset_name,
                total_samples=0,
                valid_samples=0,
                invalid_samples=0,
                errors=[f"Dataset directory not found: {dataset_dir}"],
                warnings=[],
                statistics={}
            )
        
        print(f"\nValidating dataset: {dataset_name}")
        print(f"Directory: {dataset_dir}")
        
        errors = []
        warnings = []
        statistics = {}
        
        # Find image directories
        image_dirs = []
        for split in ['train', 'val', 'test']:
            img_dir = dataset_dir / 'images' / split
            if img_dir.exists():
                image_dirs.append((split, img_dir))
        
        if not image_dirs:
            # Check for images directly in dataset directory
            image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
            if image_files:
                image_dirs.append(('all', dataset_dir))
            else:
                errors.append("No images found in dataset")
                return ValidationResult(
                    dataset_name=dataset_name,
                    total_samples=0,
                    valid_samples=0,
                    invalid_samples=0,
                    errors=errors,
                    warnings=warnings,
                    statistics=statistics
                )
        
        total_samples = 0
        valid_samples = 0
        invalid_samples = 0
        image_sizes = []
        
        # Validate images
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for split, img_dir in image_dirs:
                for img_path in img_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        futures.append(executor.submit(self.validate_image, str(img_path)))
                        total_samples += 1
            
            for future in as_completed(futures):
                is_valid, message = future.result()
                if is_valid:
                    valid_samples += 1
                else:
                    invalid_samples += 1
                    errors.append(message)
        
        # Find annotation files
        annotation_files = list(dataset_dir.glob('annotations/*.json'))
        
        for ann_file in annotation_files:
            result = self.validate_annotation_file(str(ann_file))
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Calculate statistics
        statistics = {
            'total_images': total_samples,
            'valid_images': valid_samples,
            'invalid_images': invalid_samples,
            'annotation_files': len(annotation_files),
            'splits': [split for split, _ in image_dirs],
        }
        
        # Print summary
        print(f"\nValidation Results:")
        print(f"  Total images: {total_samples}")
        print(f"  Valid images: {valid_samples}")
        print(f"  Invalid images: {invalid_samples}")
        print(f"  Success rate: {valid_samples / max(total_samples, 1) * 100:.1f}%")
        
        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"    - {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more errors")
        
        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for warning in warnings[:5]:
                print(f"    - {warning}")
        
        return ValidationResult(
            dataset_name=dataset_name,
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
    
    def validate_all_datasets(self) -> Dict[str, ValidationResult]:
        """Validate all datasets in data directory"""
        results = {}
        
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                results[dataset_dir.name] = self.validate_dataset(dataset_dir.name)
        
        return results
    
    def generate_report(self, results: Dict[str, ValidationResult], output_file: str = "validation_report.json"):
        """Generate validation report"""
        report = {
            'total_datasets': len(results),
            'valid_datasets': sum(1 for r in results.values() if r.is_valid),
            'total_samples': sum(r.total_samples for r in results.values()),
            'valid_samples': sum(r.valid_samples for r in results.values()),
            'datasets': {}
        }
        
        for name, result in results.items():
            report['datasets'][name] = {
                'total_samples': result.total_samples,
                'valid_samples': result.valid_samples,
                'success_rate': result.success_rate,
                'is_valid': result.is_valid,
                'errors': result.errors[:10],  # Limit errors in report
                'warnings': result.warnings[:10],
                'statistics': result.statistics
            }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to: {output_file}")
        return report


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IBVAP Data Validator")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dataset", help="Specific dataset to validate")
    parser.add_argument("--output", default="validation_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    
    if args.dataset:
        result = validator.validate_dataset(args.dataset)
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULT: {args.dataset}")
        print(f"{'='*60}")
        print(f"Valid: {result.is_valid}")
        print(f"Success Rate: {result.success_rate:.1f}%")
    else:
        results = validator.validate_all_datasets()
        validator.generate_report(results, args.output)


if __name__ == "__main__":
    main()
