#!/usr/bin/env bash

# This script is used in ~/.local/share/jupyter/kernels/<kernel_name>/kernel.json
# to load environment modules before starting the kernel.

# load required modules here
module purge
module load slurm
module load JupyterLab

# run the kernel
exec $@
