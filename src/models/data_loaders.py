"""
IBVAP Data Loaders
Unified data loading interface for all dataset types
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Generator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class Annotation:
    """Single annotation"""
    image_path: str
    bboxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[int]
    track_ids: Optional[List[int]] = None
    attributes: Optional[Dict] = None
    frame_idx: Optional[int] = None
    video_id: Optional[str] = None


class BaseDataset(Dataset):
    """Base dataset class for all IBVAP datasets"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.samples: List[Annotation] = []
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio"""
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img = cv2.resize(img, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = img
        
        return padded
    
    def _normalize_bbox(self, bbox: List[float], img_shape: Tuple[int, int]) -> List[float]:
        """Normalize bbox to [0, 1] range"""
        h, w = img_shape[:2]
        return [
            bbox[0] / w,
            bbox[1] / h,
            bbox[2] / w,
            bbox[3] / h
        ]


# ============================================================
# IDD Dataset Loader
# ============================================================

class IDDDataset(BaseDataset):
    """
    Indian Driving Dataset loader
    Supports both image detection and temporal sequences
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640),
        use_temporal: bool = False
    ):
        super().__init__(root_dir, split, transform, target_size)
        self.use_temporal = use_temporal
        self._load_annotations()
    
    def _load_annotations(self):
        """Load IDD annotations"""
        annotation_file = self.root_dir / f"annotations/{self.split}.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            for img_info in data.get('images', []):
                img_path = self.root_dir / f"images/{self.split}" / img_info['file_name']
                
                bboxes = []
                labels = []
                track_ids = []
                
                for ann in data.get('annotations', []):
                    if ann['image_id'] == img_info['id']:
                        bbox = ann['bbox']  # [x, y, w, h]
                        # Convert to [x1, y1, x2, y2]
                        bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        labels.append(ann['category_id'])
                        track_ids.append(ann.get('track_id', -1))
                
                self.samples.append(Annotation(
                    image_path=str(img_path),
                    bboxes=bboxes,
                    labels=labels,
                    track_ids=track_ids if track_ids else None
                ))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        annotation = self.samples[idx]
        
        # Load image
        img = self._load_image(annotation.image_path)
        
        # Resize
        img = self._resize_with_padding(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Create target
        target = {
            'bboxes': torch.tensor(annotation.bboxes, dtype=torch.float32),
            'labels': torch.tensor(annotation.labels, dtype=torch.long),
            'image_id': idx
        }
        
        if annotation.track_ids:
            target['track_ids'] = torch.tensor(annotation.track_ids, dtype=torch.long)
        
        return img, target


# ============================================================
# ANPR Dataset Loader
# ============================================================

class ANPRDataset(BaseDataset):
    """
    Automatic Number Plate Recognition dataset loader
    Supports Indian Number Plates and UFPR-ALPR
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640),
        ocr_mode: bool = False
    ):
        super().__init__(root_dir, split, transform, target_size)
        self.ocr_mode = ocr_mode
        self._load_annotations()
    
    def _load_annotations(self):
        """Load ANPR annotations"""
        annotation_file = self.root_dir / f"annotations/{self.split}.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                img_path = self.root_dir / item['image']
                
                plate_bbox = item.get('plate_bbox', [])
                plate_text = item.get('plate_text', '')
                vehicle_bbox = item.get('vehicle_bbox', [])
                
                self.samples.append(Annotation(
                    image_path=str(img_path),
                    bboxes=[plate_bbox] if plate_bbox else [],
                    labels=[0],  # 0 = license plate
                    attributes={
                        'plate_text': plate_text,
                        'vehicle_bbox': vehicle_bbox
                    }
                ))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        annotation = self.samples[idx]
        
        # Load image
        img = self._load_image(annotation.image_path)
        
        # Resize
        img = self._resize_with_padding(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Create target
        target = {
            'bboxes': torch.tensor(annotation.bboxes, dtype=torch.float32),
            'labels': torch.tensor(annotation.labels, dtype=torch.long),
            'image_id': idx
        }
        
        if annotation.attributes:
            target['plate_text'] = annotation.attributes.get('plate_text', '')
            if 'vehicle_bbox' in annotation.attributes:
                target['vehicle_bbox'] = torch.tensor(
                    annotation.attributes['vehicle_bbox'], dtype=torch.float32
                )
        
        return img, target


# ============================================================
# Surveillance Video Dataset Loader
# ============================================================

class SurveillanceDataset(Dataset):
    """
    Surveillance video dataset loader
    Supports VIRAT, MEVA, UCF-Crime
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        clip_length: int = 32,
        stride: int = 16,
        transform=None,
        target_size: Tuple[int, int] = (640, 640)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_length = clip_length
        self.stride = stride
        self.transform = transform
        self.target_size = target_size
        self.samples: List[Dict] = []
        self._load_video_list()
    
    def _load_video_list(self):
        """Load video file list"""
        video_list_file = self.root_dir / f"{self.split}_videos.txt"
        
        if video_list_file.exists():
            with open(video_list_file, 'r') as f:
                for line in f:
                    video_path = line.strip()
                    if video_path:
                        self.samples.append({
                            'video_path': str(self.root_dir / video_path),
                            'video_id': Path(video_path).stem
                        })
    
    def __len__(self) -> int:
        return len(self.samples) * 10  # Approximate clips per video
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Determine video and clip
        video_idx = idx // 10
        clip_idx = idx % 10
        
        if video_idx >= len(self.samples):
            video_idx = len(self.samples) - 1
        
        sample = self.samples[video_idx]
        
        # Open video
        cap = cv2.VideoCapture(sample['video_path'])
        
        # Calculate frame range
        start_frame = clip_idx * self.stride
        frames = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                # Pad with zeros if video ends
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        
        cap.release()
        
        # Stack frames: [T, H, W, C]
        video_tensor = np.stack(frames, axis=0)
        
        # Apply transforms
        if self.transform:
            video_tensor = self.transform(video_tensor)
        else:
            video_tensor = torch.from_numpy(video_tensor).permute(0, 3, 1, 2).float() / 255.0
        
        # Create target
        target = {
            'video_id': sample['video_id'],
            'clip_idx': clip_idx,
            'start_frame': start_frame
        }
        
        return video_tensor, target


# ============================================================
# Low-Light Dataset Loader
# ============================================================

class ExDarkDataset(BaseDataset):
    """
    ExDark dataset loader for low-light detection
    """
    
    CLASS_NAMES = [
        "bicycle", "boat", "bottle", "bus", "car", "cat", "chair",
        "cup", "dog", "motorbike", "people", "table"
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640)
    ):
        super().__init__(root_dir, split, transform, target_size)
        self._load_annotations()
    
    def _load_annotations(self):
        """Load ExDark annotations"""
        annotation_dir = self.root_dir / "annotations"
        
        if annotation_dir.exists():
            for ann_file in annotation_dir.glob("*.txt"):
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                
                img_name = lines[0].strip() if lines else None
                if not img_name:
                    continue
                
                img_path = self.root_dir / "images" / img_name
                
                bboxes = []
                labels = []
                
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        label_name = parts[0]
                        if label_name in self.CLASS_NAMES:
                            label_idx = self.CLASS_NAMES.index(label_name)
                            # Parse bbox (format varies)
                            try:
                                x, y, w, h = map(float, parts[1:5])
                                bboxes.append([x, y, x + w, y + h])
                                labels.append(label_idx)
                            except ValueError:
                                continue
                
                if bboxes:
                    self.samples.append(Annotation(
                        image_path=str(img_path),
                        bboxes=bboxes,
                        labels=labels
                    ))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        annotation = self.samples[idx]
        
        # Load image
        img = self._load_image(annotation.image_path)
        
        # Resize
        img = self._resize_with_padding(img)
        
        # Apply transforms (include low-light augmentation)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Create target
        target = {
            'bboxes': torch.tensor(annotation.bboxes, dtype=torch.float32),
            'labels': torch.tensor(annotation.labels, dtype=torch.long),
            'image_id': idx
        }
        
        return img, target


# ============================================================
# Face Detection Dataset Loader
# ============================================================

class WiderFaceDataset(BaseDataset):
    """
    WIDER FACE dataset loader for face detection
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        target_size: Tuple[int, int] = (640, 640),
        min_face_size: int = 10
    ):
        super().__init__(root_dir, split, transform, target_size)
        self.min_face_size = min_face_size
        self._load_annotations()
    
    def _load_annotations(self):
        """Load WIDER FACE annotations"""
        annotation_file = self.root_dir / "wider_face_split" / f"wider_face_{self.split}_bbgt.txt"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                img_name = lines[i].strip()
                i += 1
                
                if i >= len(lines):
                    break
                
                num_faces = int(lines[i].strip())
                i += 1
                
                bboxes = []
                for _ in range(num_faces):
                    if i >= len(lines):
                        break
                    
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        x, y, w, h = map(int, parts[:4])
                        # Filter by minimum face size
                        if w >= self.min_face_size and h >= self.min_face_size:
                            bboxes.append([x, y, x + w, y + h])
                    i += 1
                
                img_path = self.root_dir / "WIDER_" + self.split / "images" / img_name
                
                if bboxes:
                    self.samples.append(Annotation(
                        image_path=str(img_path),
                        bboxes=bboxes,
                        labels=[0] * len(bboxes)  # 0 = face
                    ))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        annotation = self.samples[idx]
        
        # Load image
        img = self._load_image(annotation.image_path)
        
        # Resize
        img = self._resize_with_padding(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Create target
        target = {
            'bboxes': torch.tensor(annotation.bboxes, dtype=torch.float32),
            'labels': torch.tensor(annotation.labels, dtype=torch.long),
            'image_id': idx
        }
        
        return img, target


# ============================================================
# Factory Functions
# ============================================================

def create_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = "train",
    **kwargs
) -> Dataset:
    """
    Factory function to create dataset by name
    
    Args:
        dataset_name: Name of the dataset
        root_dir: Root directory of the dataset
        split: Train/val/test split
        **kwargs: Additional arguments for the dataset
        
    Returns:
        Dataset instance
    """
    dataset_classes = {
        "idd": IDDDataset,
        "idd_temporal": IDDDataset,
        "indian_number_plates": ANPRDataset,
        "ufpr_alpr": ANPRDataset,
        "virat": SurveillanceDataset,
        "meva": SurveillanceDataset,
        "ucf_crime": SurveillanceDataset,
        "exdark": ExDarkDataset,
        "bdd100k": IDDDataset,
        "wider_face": WiderFaceDataset,
        "custom_border": SurveillanceDataset,
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Handle dataset-specific defaults
    if dataset_name in ["virat", "meva", "ucf_crime", "custom_border"]:
        return SurveillanceDataset(root_dir=root_dir, split=split, **kwargs)
    elif dataset_name in ["indian_number_plates", "ufpr_alpr"]:
        return ANPRDataset(root_dir=root_dir, split=split, **kwargs)
    elif dataset_name == "exdark":
        return ExDarkDataset(root_dir=root_dir, split=split, **kwargs)
    elif dataset_name == "wider_face":
        return WiderFaceDataset(root_dir=root_dir, split=split, **kwargs)
    else:
        return IDDDataset(root_dir=root_dir, split=split, **kwargs)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader from dataset
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for variable-size annotations
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Batched images and targets
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    
    return images, targets
