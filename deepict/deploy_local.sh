#!/bin/bash

export srcdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export deepict_srcdir="$( cd "$( dirname "$(dirname "$srcdir")" )/DeePiCt/3d_cnn" >/dev/null 2>&1 ; pwd -P )"
config_file=$1

# Set CUDA_VISIBLE_DEVICES to 0 if not set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=${deepict_srcdir}/src:${srcdir}
echo PYTHONPATH=$PYTHONPATH

snakemake \
    --snakefile "${srcdir}/snakefile" \
    --config config="${config_file}" gpu=$CUDA_VISIBLE_DEVICES \
    --use-conda \
    --printshellcmds \
    --cores 1 --resources gpu=1
