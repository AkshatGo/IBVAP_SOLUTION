#!/bin/bash

# IBVAP Test Runner
# This script runs all tests and generates a report

set -e

echo "=========================================="
echo "IBVAP Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Check if pytest is installed
if ! python -c "import pytest" &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing...${NC}"
    pip install pytest pytest-cov
fi

# Run tests
echo ""
echo "Running unit tests..."
echo "------------------------------------------"

python -m pytest tests/test_detector.py -v \
    --tb=short \
    --strict-markers \
    -m "not slow" \
    2>&1 | tee test_output.txt

# Check exit code
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "All tests passed!"
    echo "==========================================${NC}"
    
    # Run with coverage if requested
    if [ "$1" == "--coverage" ]; then
        echo ""
        echo "Running tests with coverage..."
        echo "------------------------------------------"
        python -m pytest tests/test_detector.py \
            --cov=src/edge \
            --cov-report=html \
            --cov-report=term-missing
        echo ""
        echo "Coverage report generated in htmlcov/"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Some tests failed!"
    echo "==========================================${NC}"
    exit 1
fi
