#!/bin/bash
# IBVAP - Start Script

echo "🛡️  IBVAP - Intelligent Border Video Analytics Platform"
echo "=================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import cv2; import numpy; import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check YOLO model
if [ ! -f "yolov8n.pt" ]; then
    echo "📥 Downloading YOLOv8 model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

echo ""
echo "🚀 Starting IBVAP Dashboard..."
echo "   Open http://localhost:8501 in your browser"
echo ""

python3 main.py dashboard
