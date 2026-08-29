# IBVAP Makefile
# Common commands for development, testing, and deployment

.PHONY: help install test lint format run-backend run-dashboard run-all docker-up docker-down clean

# Default target
help:
	@echo "IBVAP - Intelligent Border Video Analytics Platform"
	@echo "=================================================="
	@echo ""
	@echo "Development Commands:"
	@echo "  make install        Install all dependencies"
	@echo "  make install-edge   Install edge detection dependencies"
	@echo "  make install-api    Install backend API dependencies"
	@echo "  make install-dashboard  Install dashboard dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make test-fast      Run tests excluding slow tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code with black"
	@echo "  make typecheck      Run type checking with mypy"
	@echo ""
	@echo "Run Services:"
	@echo "  make run-backend    Start backend API server"
	@echo "  make run-dashboard  Start dashboard dev server"
	@echo "  make run-edge       Start edge detection module"
	@echo "  make run-all        Start all services locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up      Start all services with Docker"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-logs    View Docker logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean          Clean temporary files"
	@echo "  make demo           Run edge detection demo"
	@echo "  make verify-chain   Verify hash chain integrity"

# ============================================================
# Installation
# ============================================================

install: install-edge install-api install-dashboard
	@echo "All dependencies installed successfully!"

install-edge:
	@echo "Installing edge detection dependencies..."
	pip install -r requirements.txt

install-api:
	@echo "Backend dependencies included in main requirements.txt"

install-dashboard:
	@echo "Installing dashboard dependencies..."
	cd src/dashboard && npm install

# ============================================================
# Testing
# ============================================================

test:
	@echo "Running all tests..."
	python -m pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/ -v -m "not integration"

test-integration:
	@echo "Running integration tests..."
	python -m pytest tests/ -v -m "integration"

test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	@echo "Running fast tests..."
	python -m pytest tests/ -v -m "not slow"

test-edge:
	@echo "Running edge detection tests..."
	python -m pytest tests/test_detector.py -v

# ============================================================
# Code Quality
# ============================================================

lint:
	@echo "Running linter..."
	ruff check src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/

typecheck:
	@echo "Running type checking..."
	mypy src/ --ignore-missing-imports

# ============================================================
# Run Services
# ============================================================

run-backend:
	@echo "Starting backend API server..."
	cd src/backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	@echo "Starting dashboard dev server..."
	cd src/dashboard && npm run dev

run-edge:
	@echo "Starting edge detection module..."
	python -m src.edge.detector

run-all:
	@echo "Starting all services..."
	@make run-backend &
	@make run-dashboard &
	@echo "Services started. Backend: http://localhost:8000, Dashboard: http://localhost:3000"

# ============================================================
# Docker
# ============================================================

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

docker-restart:
	@echo "Restarting Docker services..."
	docker-compose restart

# ============================================================
# Utility
# ============================================================

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf dist/ build/ *.egg-info/
	cd src/dashboard && rm -rf node_modules/.cache

demo:
	@echo "Running edge detection demo..."
	python -c "from src.edge.detector import demo; demo()"

verify-chain:
	@echo "Verifying hash chain integrity..."
	python -c " \
		from src.edge.detector import HashChainVerifier; \
		import json; \
		events = json.load(open('data/events.json')) if __import__('os').path.exists('data/events.json') else []; \
		valid, idx = HashChainVerifier.verify_chain(events); \
		print(f'Chain valid: {valid}, Tampered index: {idx}') \
	"

# ============================================================
# Development Helpers
# ============================================================

setup-dev:
	@echo "Setting up development environment..."
	pip install -e .
	pre-commit install
	@echo "Development environment ready!"

docs:
	@echo "Generating documentation..."
	cd docs && mkdocs serve

seed-db:
	@echo "Seeding database with demo data..."
	python -c "from src.backend.main import populate_demo_data; populate_demo_data()"
