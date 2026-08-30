# IBVAP Makefile
# Common commands for development, demo, and deployment

.PHONY: help install run-server run-dashboard demo verify-chain corrupt-chain \
        docker-up docker-down docker-build docker-logs \
        convert-idd augment-anpr clean

# Default target
help:
	@echo "IBVAP - Intelligent Border Video Analytics Platform"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install Python dependencies"
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
	@echo "Dataset Prep (see docs/ROADMAP.md):"
	@echo "  make convert-idd    Convert IDD-Detection to YOLO format"
	@echo "  make augment-anpr   Augment ANPR plate dataset"
	@echo ""
	@echo "Utility:"
	@echo "  make clean          Remove caches and temporary files"

# ============================================================
# Installation
# ============================================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

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

augment-anpr:
	@echo "Augmenting ANPR plate dataset..."
	python scripts/anpr_augmentation.py \
		--images-dir data/anpr/images/train \
		--labels-dir data/anpr/labels/train \
		--out-dir data/anpr_augmented/train

# ============================================================
# Utility
# ============================================================

clean:
	@echo "Cleaning temporary files..."
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name "*.py[co]" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
