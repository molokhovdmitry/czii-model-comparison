#!/bin/bash

# Exit on error
set -e

# Update package index
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group
sudo usermod -aG docker $USER

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER && sudo systemctl restart docker

# Install NVIDIA Container Toolkit using the newer method
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

git clone https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation.git
git clone https://github.com/molokhovdmitry/czii-model-comparison.git
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/custom_train_config.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/configs
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/metric_1.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/metrics/metric_1.py
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/utils.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/utils.py
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/cfg_resnet34_custom.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/configs
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/cfg_effnetb3_custom.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/configs
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/train_segmentation.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/train_segmentation.py
cp /home/ubuntu/czii-model-comparison/competition_winner/segmentation/train.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/train.py

echo "Restart the shell to be able to run docker"
echo "Now you can run: docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.08-py3"
