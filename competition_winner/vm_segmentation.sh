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

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

exec $SHELL

git clone https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation.git
git clone https://github.com/molokhovdmitry/czii-model-comparison.git
cp /home/ubuntu/czii-model-comparison/competition_winner/common_config.py /home/ubuntu/kaggle-cryoet-1st-place-segmentation/configs

echo "Now you can run: docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.08-py3"
