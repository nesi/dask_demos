#!/usr/bin/env bash

# This script is used in ~/.local/share/jupyter/kernels/<kernel_name>/kernel.json
# to load environment modules before starting the kernel.

# load required modules here
module purge
module load slurm
module load Miniconda3/4.8.2
module load CUDA/10.2.89

# activate conda environment
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda deactivate  # enforce base environment to be unloaded
conda activate ##CONDA_VENV_PATH##

# run the kernel
exec python $@