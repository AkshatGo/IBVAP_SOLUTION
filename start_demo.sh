#!/bin/bash

# IBVAP Demo Start Script
# This script starts the complete IBVAP demo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=========================================="
echo "  IBVAP - Intelligent Border Video Analytics"
echo "  SIH 2026 Demo Launcher"
echo "=========================================="
echo -e "${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: Please run this script from the SIH2026 directory${NC}"
    exit 1
fi

# Install dependencies if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python -c "import cv2" 2>/dev/null; then
    echo -e "${YELLOW}Installing OpenCV...${NC}"
    pip install opencv-python
fi

if ! python -c "import numpy" 2>/dev/null; then
    echo -e "${YELLOW}Installing NumPy...${NC}"
    pip install numpy
fi

echo -e "${GREEN}Dependencies OK${NC}"

# Menu
echo ""
echo -e "${BLUE}Select demo mode:${NC}"
echo "1) Demo Mode (simulated camera - no webcam needed)"
echo "2) Live Mode (uses webcam)"
echo "3) Video Mode (uses video file)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting IBVAP in Demo Mode...${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 'f' for fullscreen, 'a' for alert log${NC}"
        python app.py --demo
        ;;
    2)
        echo -e "${GREEN}Starting IBVAP in Live Mode...${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 'f' for fullscreen, 'a' for alert log${NC}"
        python app.py --no-demo
        ;;
    3)
        read -p "Enter video file path: " video_path
        if [ ! -f "$video_path" ]; then
            echo -e "${RED}Error: Video file not found${NC}"
            exit 1
        fi
        echo -e "${GREEN}Starting IBVAP with video: $video_path${NC}"
        echo -e "${YELLOW}Press 'q' to quit, 'f' for fullscreen, 'a' for alert log${NC}"
        python app.py --video "$video_path"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Demo complete!${NC}"
