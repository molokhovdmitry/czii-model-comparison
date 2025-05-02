#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for CryoET training..."

# Install required system packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip git cmake build-essential python3-dev zlib1g-dev

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install CUDA dependencies
echo "Installing CUDA dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone repositories
echo "Cloning repositories..."
git clone https://github.com/BloodAxe/Kaggle-2024-CryoET
git clone https://github.com/molokhovdmitry/czii-model-comparison

# Install other requirements
echo "Installing Python dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install -r czii-model-comparison/competition_winner/requirements.txt

# Create directory for pretrained weights
echo "Creating directory for pretrained weights..."
mkdir -p pretrained/wholeBody_ct_segmentation/models

echo "Downloading and setting up pretrained weights..."
wget https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/wholeBody_ct_segmentation_v0.1.9.zip
unzip wholeBody_ct_segmentation_v0.1.9.zip -d pretrained/wholeBody_ct_segmentation/models/
mv pretrained/wholeBody_ct_segmentation/models/wholeBody_ct_segmentation_v0.1.9/model.pt pretrained/wholeBody_ct_segmentation/models/
rm -rf pretrained/wholeBody_ct_segmentation/models/wholeBody_ct_segmentation_v0.1.9
rm wholeBody_ct_segmentation_v0.1.9.zip

cp /home/ubuntu/czii-model-comparison/competition_winner/train_object_detection.sh /home/ubuntu/Kaggle-2024-CryoET/train.sh

echo "Setup complete! To start training:"
echo "Run: bash vkr/Kaggle-2024-CryoET/train.sh"
