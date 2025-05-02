#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for CryoET training..."

# Install required system packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install CUDA dependencies
echo "Installing CUDA dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing Python dependencies..."
pip install -r vkr/Kaggle-2024-CryoET/requirements.txt

# Create directory for pretrained weights
echo "Creating directory for pretrained weights..."
mkdir -p pretrained/wholeBody_ct_segmentation/models

echo "Setup complete! To start training:"
echo "1. Download pretrained weights from: https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/wholeBody_ct_segmentation_v0.1.9.zip"
echo "2. Unzip the model checkpoint to pretrained/wholeBody_ct_segmentation/models/model.pt"
echo "3. Run: bash vkr/Kaggle-2024-CryoET/train.sh"
