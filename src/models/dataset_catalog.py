"""
IBVAP Dataset Catalog
Central configuration for all training datasets
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from pathlib import Path


class DatasetCategory(Enum):
    """Dataset categories"""
    VEHICLE_DETECTION = "vehicle_detection"
    TEMPORAL_TRACKING = "temporal_tracking"
    ANPR = "anpr"
    SURVEILLANCE = "surveillance"
    LOW_LIGHT = "low_light"
    FACE_DETECTION = "face_detection"
    ANOMALY = "anomaly"
    CUSTOM = "custom"


class Modality(Enum):
    """Data modality"""
    IMAGE = "image"
    VIDEO = "video"
    SEQUENCE = "sequence"


@dataclass
class DatasetInfo:
    """Information about a single dataset"""
    name: str
    short_name: str
    category: DatasetCategory
    modality: Modality
    url: str
    scale: str
    annotations: str
    purpose: str
    pipeline_module: str
    download_url: Optional[str] = None
    license: str = "Unknown"
    requires_registration: bool = False
    estimated_size_gb: float = 0.0
    classes: List[str] = field(default_factory=list)


# ============================================================
# Dataset Catalog
# ============================================================

DATASETS: Dict[str, DatasetInfo] = {
    
    # ============================================================
    # 1. Indian Context & Heterogeneous Vehicle/Traffic Analytics
    # ============================================================
    
    "idd": DatasetInfo(
        name="Indian Driving Dataset",
        short_name="IDD",
        category=DatasetCategory.VEHICLE_DETECTION,
        modality=Modality.IMAGE,
        url="https://iith.ac.in/projects/idddataset/",
        download_url="https://www.kaggle.com/datasets/insaan/indian-driving-dataset",
        scale="10,000 finely annotated images (34 classes); IDD Detection: 40,000 bounding-box images",
        annotations="Bounding boxes + semantic segmentation masks",
        purpose="Fine-tuning base detectors for Indian road scenarios, regional vehicle types, pedestrian patterns",
        pipeline_module="detection",
        license="Academic Use",
        estimated_size_gb=2.5,
        classes=[
            "auto_rickshaw", "bus", "car", "motorcycle", "truck", "bicycle",
            "person", "traffic_light", "traffic_sign", "road_divider",
            "lane_divider", "parking", "rider", "trailer", "vehicle_group",
            "two_wheeler", "three_wheeler", "construction_vehicle"
        ]
    ),
    
    "idd_temporal": DatasetInfo(
        name="IDD Temporal",
        short_name="IDD-Temporal",
        category=DatasetCategory.TEMPORAL_TRACKING,
        modality=Modality.SEQUENCE,
        url="https://idd.insaan.co.in/",
        scale="Sequential frames (±15 frames around key IDD frames)",
        annotations="Temporal bounding boxes + track IDs",
        purpose="Evaluating temporal consistency, frame-to-frame tracking, localized motion models",
        pipeline_module="tracking",
        license="Academic Use",
        estimated_size_gb=5.0
    ),
    
    "iitm_hetra": DatasetInfo(
        name="IITM-HeTra Video Surveillance Dataset",
        short_name="IITM-HeTra",
        category=DatasetCategory.SURVEILLANCE,
        modality=Modality.VIDEO,
        url="https://datasetninja.com/iitm-hetra",
        download_url="https://github.com/suriya-iitm/HeTra",
        scale="1,400+ annotated video frames from overhead Chennai traffic feeds",
        annotations="Bounding boxes + track IDs",
        purpose="Stationary overhead CCTV validation for vehicle counting, classification, pedestrian tracking",
        pipeline_module="tracking",
        license="Academic Use",
        estimated_size_gb=1.2
    ),
    
    # ============================================================
    # 2. ANPR Datasets
    # ============================================================
    
    "indian_number_plates": DatasetInfo(
        name="Indian Number Plates Dataset",
        short_name="INP",
        category=DatasetCategory.ANPR,
        modality=Modality.IMAGE,
        url="https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset",
        download_url="https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset",
        scale="15,000+ HD images (10,000 with annotations)",
        annotations="Bounding boxes + character-level annotations",
        purpose="Localized ANPR and OCR pipeline development for Indian plates",
        pipeline_module="anpr",
        license="CC0 / Open Data Commons",
        estimated_size_gb=3.0,
        classes=[
            "license_plate", "vehicle", "two_wheeler", "four_wheeler",
            "commercial_truck", "auto_rickshaw"
        ]
    ),
    
    "ufpr_alpr": DatasetInfo(
        name="UFPR-ALPR Dataset",
        short_name="UFPR-ALPR",
        category=DatasetCategory.ANPR,
        modality=Modality.IMAGE,
        url="https://www.kaggle.com/datasets/andrewmvd/ufpr-alpr",
        download_url="https://www.kaggle.com/datasets/andrewmvd/ufpr-alpr",
        scale="4,500 Full HD images (1920×1080) across 150 camera angles",
        annotations="Bounding boxes + plate characters",
        purpose="Standard benchmark for license plate localization and OCR under varying illumination",
        pipeline_module="anpr",
        license="Academic Use",
        estimated_size_gb=1.5
    ),
    
    # ============================================================
    # 3. Surveillance & Activity Recognition
    # ============================================================
    
    "virat": DatasetInfo(
        name="VIRAT Video Surveillance Dataset",
        short_name="VIRAT",
        category=DatasetCategory.SURVEILLANCE,
        modality=Modality.VIDEO,
        url="https://viratdata.org/",
        download_url="https://viratdata.org/",
        scale="8.5+ hours of HD stationary outdoor surveillance video",
        annotations="Event annotations + bounding boxes",
        purpose="Virtual fence intrusion detection, loitering detection, action-based threat heuristics",
        pipeline_module="surveillance",
        license="Academic Use",
        estimated_size_gb=50.0,
        requires_registration=True
    ),
    
    "meva": DatasetInfo(
        name="MEVA (Multiview Extended Video with Activities)",
        short_name="MEVA",
        category=DatasetCategory.SURVEILLANCE,
        modality=Modality.VIDEO,
        url="http://mevadata.org/",
        download_url="http://mevadata.org/",
        scale="~330 hours across 4,001 clips (~470 GB)",
        annotations="Activity annotations + bounding boxes",
        purpose="Multi-camera activity detection, cross-camera tracking, facility-wide monitoring",
        pipeline_module="cross_camera",
        license="Academic Use (KFD1)",
        estimated_size_gb=470.0,
        requires_registration=True
    ),
    
    "ucf_crime": DatasetInfo(
        name="UCF-Crime Dataset",
        short_name="UCF-Crime",
        category=DatasetCategory.ANOMALY,
        modality=Modality.VIDEO,
        url="https://www.crcv.ucf.edu/data/UCF-Crime.php",
        download_url="https://www.kaggle.com/datasets/viditv/ucf-crime-dataset",
        scale="128 hours across 1,900 untrimmed CCTV sequences (13 anomaly categories)",
        annotations="Video-level anomaly labels",
        purpose="Anomaly classification for automated high-priority alert workflows",
        pipeline_module="anomaly_detection",
        license="Academic Use",
        estimated_size_gb=15.0,
        classes=[
            "abuse", "arrest", "arson", "assault", "burglary", "explosion",
            "fighting", "normal", "road_accidents", "robbery", "shooting",
            "stealing", "vandalism"
        ]
    ),
    
    # ============================================================
    # 4. Low-Light Vision
    # ============================================================
    
    "exdark": DatasetInfo(
        name="ExDark (Exclusively Dark Image Dataset)",
        short_name="ExDark",
        category=DatasetCategory.LOW_LIGHT,
        modality=Modality.IMAGE,
        url="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        download_url="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        scale="7,363 images across 12 object classes",
        annotations="Bounding boxes + object classes",
        purpose="Low-light object detection robustness and night patrol risk escalation",
        pipeline_module="low_light",
        license="Academic Use",
        estimated_size_gb=0.8,
        classes=[
            "bicycle", "boat", "bottle", "bus", "car", "cat", "chair",
            "cup", "dog", "motorbike", "people", "table"
        ]
    ),
    
    # ============================================================
    # 5. Secondary Benchmarks
    # ============================================================
    
    "bdd100k": DatasetInfo(
        name="BDD100K",
        short_name="BDD100K",
        category=DatasetCategory.VEHICLE_DETECTION,
        modality=Modality.VIDEO,
        url="https://bdd-data.berkeley.edu/",
        download_url="https://bdd-data.berkeley.edu/",
        scale="100,000 US driving videos (~2,000 tracking sequences, 130,600+ track IDs)",
        annotations="Bounding boxes + tracking IDs + attributes",
        purpose="Secondary benchmark for vehicle tracking and adverse-weather perception",
        pipeline_module="tracking",
        license="Creative Commons",
        estimated_size_gb=100.0,
        requires_registration=True
    ),
    
    # ============================================================
    # 6. Face Detection
    # ============================================================
    
    "wider_face": DatasetInfo(
        name="WIDER FACE Dataset",
        short_name="WIDER FACE",
        category=DatasetCategory.FACE_DETECTION,
        modality=Modality.IMAGE,
        url="https://shuoyang1213.me/WIDERFACE/",
        download_url="https://shuoyang1213.me/WIDERFACE/",
        scale="32,203 images containing 393,703 annotated faces",
        annotations="Bounding boxes with attributes (blur, illumination, invalid, occlusion, pose)",
        purpose="Scale-invariant face detection and quality filtering",
        pipeline_module="face_detection",
        license="Academic Use",
        estimated_size_gb=3.5
    ),
    
    # ============================================================
    # 7. Custom Border Dataset
    # ============================================================
    
    "custom_border": DatasetInfo(
        name="Custom Border Surveillance Dataset",
        short_name="Border-Custom",
        category=DatasetCategory.CUSTOM,
        modality=Modality.VIDEO,
        url="Local dataset (to be collected)",
        scale="TBD - Border crossing, perimeter intrusion, loitering, vehicle intrusion clips",
        annotations="To be annotated",
        purpose="Domain fine-tuning and operational validation on actual border scenarios",
        pipeline_module="full_pipeline",
        license="Internal Use Only",
        estimated_size_gb=10.0,
        classes=[
            "boundary_crossing", "perimeter_intrusion", "loitering",
            "suspicious_movement", "vehicle_intrusion", "night_activity",
            "person", "vehicle", "animal", "patrol"
        ]
    ),
}


# ============================================================
# Dataset Groupings
# ============================================================

DETECTION_DATASETS = ["idd", "bdd100k"]
TRACKING_DATASETS = ["idd_temporal", "iitm_hetra", "bdd100k"]
ANPR_DATASETS = ["indian_number_plates", "ufpr_alpr"]
SURVEILLANCE_DATASETS = ["virat", "meva", "ucf_crime"]
LOW_LIGHT_DATASETS = ["exdark"]
FACE_DATASETS = ["wider_face"]
CUSTOM_DATASETS = ["custom_border"]

# Quick reference for common training combinations
TRAINING_CONFIGS = {
    "detection_only": DETECTION_DATASETS,
    "anpr_only": ANPR_DATASETS,
    "surveillance_only": SURVEILLANCE_DATASETS,
    "full_pipeline": DETECTION_DATASETS + TRACKING_DATASETS + ANPR_DATASETS + SURVEILLANCE_DATASETS,
    "lightweight": ["idd", "indian_number_plates", "exdark"],
    "complete": list(DATASETS.keys()),
}


def get_dataset(name: str) -> DatasetInfo:
    """Get dataset info by name"""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def list_datasets(category: DatasetCategory = None) -> List[str]:
    """List all datasets, optionally filtered by category"""
    if category:
        return [k for k, v in DATASETS.items() if v.category == category]
    return list(DATASETS.keys())


def get_total_size_gb() -> float:
    """Get total estimated size of all datasets"""
    return sum(d.estimated_size_gb for d in DATASETS.values())


def print_dataset_summary():
    """Print summary of all datasets"""
    print("\n" + "=" * 80)
    print("IBVAP DATASET CATALOG SUMMARY")
    print("=" * 80)
    
    for category in DatasetCategory:
        datasets = list_datasets(category)
        if datasets:
            print(f"\n{category.value.upper().replace('_', ' ')}:")
            print("-" * 60)
            for name in datasets:
                d = DATASETS[name]
                print(f"  {d.short_name:15} | {d.modality.value:10} | {d.estimated_size_gb:6.1f} GB | {d.scale[:50]}")
    
    print("\n" + "=" * 80)
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Total estimated size: {get_total_size_gb():.1f} GB")
    print("=" * 80)


if __name__ == "__main__":
    print_dataset_summary()
