#!/bin/bash
#
# Pod environment setup script for basketball video analysis
# Run this on the pod after initial SSH connection to configure environment
#
# Usage:
#   bash pod_setup.sh
#

set -e  # Exit on error

echo "========================================"
echo "Pod Environment Setup"
echo "========================================"
echo ""

# Check Python version
echo "[1/8] Checking Python installation..."
python3 --version
python3 -c "import sys; print(f'  Python path: {sys.executable}')"
echo ""

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "[2/8] Creating Python virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "[2/8] Virtual environment already exists"
fi

# Activate virtual environment
echo "[3/8] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[4/8] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "  ✓ pip upgraded"
echo ""

# Install PyTorch with CUDA support
echo "[5/8] Installing PyTorch with CUDA 12.1 support..."
echo "  (This may take 2-3 minutes)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "  ✓ PyTorch installed"
echo ""

# Install YOLOv8
echo "[6/8] Installing YOLOv8..."
pip install ultralytics
echo "  ✓ YOLOv8 installed"
echo ""

# Install other dependencies
echo "[7/8] Installing other dependencies..."
pip install opencv-python numpy pandas pyyaml scikit-learn scipy
echo "  ✓ Dependencies installed"
echo ""

# Verify CUDA
echo "[8/8] Verifying CUDA and GPU access..."
python3 << 'EOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  WARNING: CUDA not available - GPU will not be used!")
EOF
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Create /data directories for videos:"
echo "     mkdir -p /data/videos"
echo "     mkdir -p /data/runs"
echo ""
echo "  2. Upload videos via SCP from your machine:"
echo "     scp mac9.mp4 user@pod_ip:/data/videos/"
echo ""
echo "  3. Run batch processing:"
echo "     python process_batch.py --dir /data/videos/ --output /data/runs/"
echo ""
echo "  4. Download results back to your machine:"
echo "     scp -r user@pod_ip:/data/runs/run_* ."
echo ""
echo "  5. Validate with QC checker:"
echo "     python qc_checker.py runs/run_*/"
echo ""
