#!/bin/bash
set -e

mkdir czii-cryo-et-object-identification

git clone https://github.com/ZauggGroup/DeePiCt.git
git clone https://github.com/molokhovdmitry/czii-model-comparison.git
cp /home/ubuntu/czii-model-comparison/deepict/particle_picking_evaluation.py DeePiCt/3d_cnn/scripts/particle_picking_evaluation.py
cp /home/ubuntu/czii-model-comparison/deepict/statistics_utils.py DeePiCt/3d_cnn/src/performance/statistics_utils.py
cd DeePiCt

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh

source $HOME/miniconda3/etc/profile.d/conda.sh

echo 'source $HOME/miniconda3/bin/activate' >> ~/.bashrc

conda install -n base -c conda-forge mamba -y

mamba create -c conda-forge -c bioconda -n snakemake snakemake==5.13.0 python=3.7 -y
conda activate snakemake

conda install pandas -y

conda install -c pytorch pytorch-gpu torchvision -y

conda install -c conda-forge keras-gpu=2.3.1 -y

echo "DeePiCt installation completed successfully!"
echo "To use DeePiCt, activate the environment with: conda activate snakemake"
echo "For 2D CNN pipeline: bash /path/to/2d_cnn/deploy_cluster.sh /path/to/config.yaml"
echo "For 3D CNN pipeline: bash /path/to/3d_cnn/deploy_cluster.sh /path/to/config.yaml"
echo "For local deployment use deploy_local.sh instead of deploy_cluster.sh"

mkdir -p /home/ubuntu/czii-cryo-et-object-identification

source $HOME/miniconda3/bin/activate
pip install dotenv zarr emfile mrcfile


