"""
IBVAP Data Transforms
Preprocessing and augmentation pipeline for all dataset types
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Image Transforms
# ============================================================

class IBVAPTransforms:
    """
    Unified transform pipeline for IBVAP datasets
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.input_size = input_size
        self.augment = augment
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Build transforms
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        self.test_transform = self._build_test_transform()
    
    def _build_train_transform(self) -> A.Compose:
        """Build training augmentation pipeline"""
        transform_list = [
            A.Resize(self.input_size[1], self.input_size[0]),
        ]
        
        if self.augment:
            transform_list.extend([
                # Spatial transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5),
                ], p=0.3),
                
                # Color transforms
                A.OneOf([
                    A.CLAHE(clip_limit=2.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.RandomGamma(),
                ], p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                
                # Noise transforms
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                ], p=0.2),
                
                # Low-light augmentation (for ExDark training)
                A.RandomGamma(gamma_limit=(60, 140), p=0.2),
                A.CLAHE(clip_limit=4.0, p=0.2),
            ])
        
        # Always resize and normalize
        transform_list.extend([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])
        
        return A.Compose(
            transform_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            )
        )
    
    def _build_val_transform(self) -> A.Compose:
        """Build validation transform pipeline (no augmentation)"""
        return A.Compose([
            A.Resize(self.input_size[1], self.input_size[0]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def _build_test_transform(self) -> A.Compose:
        """Build test transform pipeline"""
        return self.val_transform
    
    def __call__(self, image: np.ndarray, bboxes: List = None, labels: List = None):
        """Apply transforms to image and annotations"""
        if bboxes is not None and labels is not None:
            # Training/Validation with annotations
            if self.augment:
                result = self.train_transform(
                    image=image,
                    bboxes=bboxes,
                    labels=labels
                )
            else:
                result = self.val_transform(
                    image=image,
                    bboxes=bboxes,
                    labels=labels
                )
            return result['image'], result['bboxes'], result['labels']
        else:
            # Test time (no annotations)
            result = self.test_transform(image=image)
            return result['image']


# ============================================================
# Specialized Transforms
# ============================================================

class LowLightTransforms:
    """
    Specialized transforms for low-light scenarios
    Used for ExDark and night-time surveillance training
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size
        self.transform = A.Compose([
            A.Resize(input_size[1], input_size[0]),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.2),
                    contrast_limit=(-0.4, 0.2),
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(40, 100), p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.8),
            A.GaussNoise(var_limit=(10, 80), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def __call__(self, image: np.ndarray, bboxes: List = None, labels: List = None):
        if bboxes is not None:
            result = self.transform(image=image, bboxes=bboxes, labels=labels)
            return result['image'], result['bboxes'], result['labels']
        else:
            result = self.transform(image=image)
            return result['image']


class ANPRTransforms:
    """
    Specialized transforms for ANPR (license plate recognition)
    Focuses on preserving plate text clarity
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size
        self.transform = A.Compose([
            A.Resize(input_size[1], input_size[0]),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ))
    
    def __call__(self, image: np.ndarray, bboxes: List = None, labels: List = None):
        if bboxes is not None:
            result = self.transform(image=image, bboxes=bboxes, labels=labels)
            return result['image'], result['bboxes'], result['labels']
        else:
            result = self.transform(image=image)
            return result['image']


class SurveillanceTransforms:
    """
    Specialized transforms for surveillance video
    Preserves temporal consistency
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size
        self.spatial_transform = A.Compose([
            A.Resize(input_size[1], input_size[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def __call__(self, video: np.ndarray):
        """
        Apply transforms to video clip
        
        Args:
            video: Video tensor of shape [T, H, W, C]
            
        Returns:
            Transformed video tensor
        """
        # Apply same spatial transform to all frames
        transformed_frames = []
        for frame in video:
            result = self.spatial_transform(image=frame)
            transformed_frames.append(result['image'])
        
        return torch.stack(transformed_frames, dim=0)


# ============================================================
# Video Augmentation
# ============================================================

class VideoAugmentation:
    """
    Temporal-aware video augmentation
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, video: np.ndarray) -> np.ndarray:
        """
        Apply consistent augmentation across video frames
        
        Args:
            video: Video array of shape [T, H, W, C]
            
        Returns:
            Augmented video
        """
        if np.random.random() > self.p:
            return video
        
        # Random horizontal flip (consistent across frames)
        if np.random.random() > 0.5:
            video = np.flip(video, axis=2).copy()
        
        # Random brightness (consistent across frames)
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            video = np.clip(video * brightness, 0, 255).astype(np.uint8)
        
        return video


# ============================================================
# Factory Functions
# ============================================================

def get_transforms(
    dataset_type: str,
    input_size: Tuple[int, int] = (640, 640),
    augment: bool = True
) -> IBVAPTransforms:
    """
    Get appropriate transforms for dataset type
    
    Args:
        dataset_type: Type of dataset
        input_size: Input image size
        augment: Whether to apply augmentation
        
    Returns:
        Transform pipeline
    """
    if dataset_type == "exdark":
        return LowLightTransforms(input_size)
    elif dataset_type in ["indian_number_plates", "ufpr_alpr"]:
        return ANPRTransforms(input_size)
    elif dataset_type in ["virat", "meva", "ucf_crime", "custom_border"]:
        return SurveillanceTransforms(input_size)
    else:
        return IBVAPTransforms(input_size, augment=augment)


# ============================================================
# Test Transforms
# ============================================================

if __name__ == "__main__":
    # Test transforms
    import matplotlib.pyplot as plt
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200], [300, 150, 400, 250]]
    labels = [0, 1]
    
    # Test each transform type
    transform_types = ["idd", "exdark", "anpr", "surveillance"]
    
    for transform_type in transform_types:
        print(f"\nTesting {transform_type} transforms...")
        transform = get_transforms(transform_type, augment=True)
        
        if isinstance(transform, SurveillanceTransforms):
            # Video transform
            video = np.random.randint(0, 255, (32, 480, 640, 3), dtype=np.uint8)
            result = transform(video)
            print(f"  Video shape: {result.shape}")
        else:
            # Image transform
            result_img, result_bboxes, result_labels = transform(image, bboxes, labels)
            print(f"  Image shape: {result_img.shape}")
            print(f"  Bboxes: {result_bboxes}")
            print(f"  Labels: {result_labels}")
    
    print("\nAll transforms tested successfully!")
