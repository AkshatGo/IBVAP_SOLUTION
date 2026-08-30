# IBVAP Makefile
# Common commands for development, demo, and deployment

.PHONY: help install install-core test run-server run-dashboard demo verify-chain corrupt-chain \
        docker-up docker-down docker-build docker-logs \
        convert-idd convert-exdark darken-detection convert-plates augment-anpr \
        train-detection train-plate eval-detection eval-anpr eval-threshold \
        export-onnx clean

# Default target
help:
	@echo "IBVAP - Intelligent Border Video Analytics Platform"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install all dependencies (core + ML)"
	@echo "  make install-core   Core only — what Streamlit Cloud installs"
	@echo "  make test           Run the test suite (no models downloaded)"
	@echo ""
	@echo "Run Services:"
	@echo "  make run-server     Start FastAPI backend (port 8000)"
	@echo "  make run-dashboard  Start Streamlit dashboard (port 8501)"
	@echo "  make demo           Run edge pipeline on webcam/video"
	@echo ""
	@echo "Hash Chain:"
	@echo "  make verify-chain   Verify tamper-evident chain integrity"
	@echo "  make corrupt-chain  Corrupt one record (demo: shows chain break)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-build   Build images"
	@echo "  make docker-logs    Tail logs"
	@echo ""
	@echo "Dataset Prep (see docs/ROADMAP.md §3):"
	@echo "  make convert-idd       Convert IDD-Detection to YOLO format"
	@echo "  make convert-exdark    Convert ExDark (low-light) to YOLO format"
	@echo "  make darken-detection  Bake synthetic night copies of train images"
	@echo "  make convert-plates    Convert plate datasets to single-class YOLO"
	@echo "  make augment-anpr      Augment the ANPR plate dataset"
	@echo ""
	@echo "Training & Evaluation (needs a GPU — see docs/ROADMAP.md §4):"
	@echo "  make train-detection   Fine-tune the person/vehicle detector"
	@echo "  make train-plate       Fine-tune the plate localizer"
	@echo "  make eval-detection    mAP with a day/night breakout"
	@echo "  make eval-anpr         End-to-end plate accuracy"
	@echo "  make eval-threshold    Re-validate the confidence cutoff"
	@echo "  make export-onnx       Export weights to ONNX for edge deployment"
	@echo ""
	@echo "Utility:"
	@echo "  make clean          Remove caches and temporary files"

# ============================================================
# Installation
# ============================================================

install:
	@echo "Installing all dependencies..."
	pip install -r requirements-ml.txt

install-core:
	@echo "Installing core deps only (exactly what Streamlit Cloud installs)..."
	pip install -r requirements.txt

test:
	@echo "Running test suite..."
	python -m pytest

# ============================================================
# Run Services
# ============================================================

run-server:
	@echo "Starting IBVAP API server on http://localhost:8000 ..."
	python main.py server --host 0.0.0.0 --port 8000

run-dashboard:
	@echo "Starting IBVAP dashboard on http://localhost:8501 ..."
	python main.py dashboard

demo:
	@echo "Running edge pipeline demo..."
	python main.py demo

# ============================================================
# Hash Chain
# ============================================================

verify-chain:
	@echo "Verifying hash chain integrity..."
	python -m src.edge.hashchain verify

corrupt-chain:
	@echo "Corrupting one chain record (demo only)..."
	python -m src.edge.hashchain corrupt

# ============================================================
# Docker
# ============================================================

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

# ============================================================
# Dataset Preparation
# ============================================================

convert-idd:
	@echo "Converting IDD-Detection to YOLO format..."
	python scripts/idd_to_yolo.py --idd-root data/idd_detection --out-root data/detection

convert-exdark:
	@echo "Converting ExDark to YOLO format..."
	python scripts/exdark_to_yolo.py convert --exdark-root data/ExDark --out-root data/detection

darken-detection:
	@echo "Synthesising low-light training copies..."
	python scripts/exdark_to_yolo.py darken --out-root data/detection --fraction 0.30

convert-plates:
	@echo "Converting plate datasets to YOLO format..."
	python scripts/plates_to_yolo.py --root data/raw/car-plate-detection --format voc
	python scripts/plates_to_yolo.py --root data/raw/UFPR-ALPR --format ufpr

augment-anpr:
	@echo "Augmenting ANPR plate dataset..."
	python scripts/anpr_augmentation.py \
		--images-dir data/anpr/images/train \
		--labels-dir data/anpr/labels/train \
		--out-dir data/anpr_augmented/train

# ============================================================
# Training & Evaluation
# ============================================================

DETECTION_WEIGHTS ?= runs/detect/ibvap_detection/weights/best.pt
PLATE_WEIGHTS     ?= runs/detect/ibvap_plate/weights/best.pt

train-detection:
	python scripts/train.py detection --data data/detection/data.yaml

train-plate:
	python scripts/train.py plate --data data/anpr/data.yaml

eval-detection:
	python scripts/evaluate.py detection --weights $(DETECTION_WEIGHTS) \
		--data data/detection/data.yaml

eval-anpr:
	python scripts/evaluate.py anpr --images-dir data/anpr/images/val \
		--ground-truth data/anpr/plates_val.json --plate-model $(PLATE_WEIGHTS)

eval-threshold:
	python scripts/evaluate.py threshold --weights $(DETECTION_WEIGHTS) \
		--data data/detection/data.yaml

export-onnx:
	python scripts/export_onnx.py --weights $(DETECTION_WEIGHTS)

# ============================================================
# Utility
# ============================================================

clean:
	@echo "Cleaning temporary files..."
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name "*.py[co]" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
