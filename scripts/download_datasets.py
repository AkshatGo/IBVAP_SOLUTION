"""
IBVAP Dataset Downloader
Download and prepare all training datasets
"""

import os
import sys
import json
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class DatasetDownloader:
    """
    Download and prepare datasets for IBVAP training
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download URLs (these are examples - replace with actual URLs)
        self.datasets = {
            "indian_number_plates": {
                "url": "https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset",
                "type": "kaggle",
                "requires_auth": True,
                "size_gb": 3.0
            },
            "exdark": {
                "url": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
                "type": "git",
                "requires_auth": False,
                "size_gb": 0.8
            },
            "wider_face": {
                "url": "https://shuoyang1213.me/WIDERFACE/",
                "type": "direct",
                "requires_auth": True,
                "size_gb": 3.5
            },
            "idd": {
                "url": "https://iith.ac.in/projects/idddataset/",
                "type": "direct",
                "requires_auth": True,
                "size_gb": 2.5
            },
            "virat": {
                "url": "https://viratdata.org/",
                "type": "registration",
                "requires_auth": True,
                "size_gb": 50.0
            },
            "ucf_crime": {
                "url": "https://www.crcv.ucf.edu/data/UCF-Crime.php",
                "type": "direct",
                "requires_auth": False,
                "size_gb": 15.0
            },
        }
    
    def download_file(self, url: str, dest_path: Path, description: str = "") -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, dest_dir: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(dest_dir)
            else:
                print(f"Unknown archive format: {archive_path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Extraction failed: {e}")
            return False
    
    def clone_git_repo(self, url: str, dest_dir: Path) -> bool:
        """Clone a git repository"""
        try:
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest_dir)],
                check=True,
                capture_output=True
            )
            return True
        except Exception as e:
            print(f"Git clone failed: {e}")
            return False
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download a specific dataset"""
        if dataset_name not in self.datasets:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        dest_dir = self.data_dir / dataset_name
        
        # Check if already downloaded
        if dest_dir.exists() and not force:
            print(f"Dataset {dataset_name} already exists at {dest_dir}")
            print("Use --force to re-download")
            return True
        
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_name}")
        print(f"URL: {dataset_info['url']}")
        print(f"Estimated size: {dataset_info['size_gb']:.1f} GB")
        print(f"{'='*60}\n")
        
        # Handle different download types
        if dataset_info['type'] == 'kaggle':
            print("KAGGLE DATASET")
            print("Please download manually from Kaggle:")
            print(f"  {dataset_info['url']}")
            print(f"\nThen extract to: {dest_dir}")
            print("Or use: kaggle datasets download -d <dataset-slug> -p data/")
            return False
        
        elif dataset_info['type'] == 'git':
            print("Cloning git repository...")
            return self.clone_git_repo(dataset_info['url'], dest_dir)
        
        elif dataset_info['type'] == 'registration':
            print("REGISTRATION REQUIRED")
            print("Please register and download from:")
            print(f"  {dataset_info['url']}")
            print(f"\nThen extract to: {dest_dir}")
            return False
        
        elif dataset_info['type'] == 'direct':
            print("DIRECT DOWNLOAD")
            print("Please download manually from:")
            print(f"  {dataset_info['url']}")
            print(f"\nThen extract to: {dest_dir}")
            return False
        
        return False
    
    def prepare_dataset(self, dataset_name: str) -> bool:
        """Prepare a dataset for training"""
        dest_dir = self.data_dir / dataset_name
        
        if not dest_dir.exists():
            print(f"Dataset {dataset_name} not found at {dest_dir}")
            return False
        
        print(f"\nPreparing dataset: {dataset_name}")
        
        # Create standard directory structure
        (dest_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dest_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dest_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (dest_dir / "annotations").mkdir(parents=True, exist_ok=True)
        
        # Create dataset info file
        info = {
            "name": dataset_name,
            "root_dir": str(dest_dir),
            "prepared": True,
            "splits": ["train", "val", "test"]
        }
        
        with open(dest_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Dataset prepared at: {dest_dir}")
        return True
    
    def download_all(self, datasets: Optional[List[str]] = None, force: bool = False):
        """Download all or specified datasets"""
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        print("=" * 60)
        print("IBVAP Dataset Downloader")
        print("=" * 60)
        print(f"\nDownloading {len(datasets)} datasets to: {self.data_dir}")
        
        results = {}
        for dataset_name in datasets:
            print(f"\n{'-'*60}")
            results[dataset_name] = self.download_dataset(dataset_name, force)
        
        # Summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        
        print("\n" + "=" * 60)
        
        return results
    
    def print_manual_download_guide(self):
        """Print manual download instructions"""
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD GUIDE")
        print("=" * 60)
        
        print("""
For datasets that require manual download, follow these steps:

1. Indian Number Plates (Kaggle):
   - Install Kaggle CLI: pip install kaggle
   - Set up API key: kaggle datasets download -d dataclusterlabs/indian-number-plates-dataset -p data/

2. IDD (IIT Hyderabad):
   - Visit: https://iith.ac.in/projects/idddataset/
   - Register and download
   - Extract to data/idd/

3. WIDER FACE:
   - Visit: https://shuoyang1213.me/WIDERFACE/
   - Download images and annotations
   - Extract to data/wider_face/

4. VIRAT:
   - Visit: https://viratdata.org/
   - Register for access
   - Download and extract to data/virat/

5. UFPR-ALPR:
   - Visit: https://www.kaggle.com/datasets/andrewmvd/ufpr-alpr
   - Download and extract to data/ufpr_alpr/

6. BDD100K:
   - Visit: https://bdd-data.berkeley.edu/
   - Register and download
   - Extract to data/bdd100k/

After downloading, run:
   python scripts/download_datasets.py --prepare <dataset_name>
""")


def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Downloader")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--prepare", nargs="+", help="Prepare downloaded datasets")
    parser.add_argument("--guide", action="store_true", help="Print manual download guide")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.guide:
        downloader.print_manual_download_guide()
    elif args.prepare:
        for dataset in args.prepare:
            downloader.prepare_dataset(dataset)
    else:
        downloader.download_all(args.datasets, args.force)


if __name__ == "__main__":
    main()
