#!/usr/bin/env bash
#SBATCH --account=nesi99999
#SBATCH --partition=milan
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

# load environment modules and activate conda environment
module purge && module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

conda deactivate
conda activate ./venv

# increase reliability of connections
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=3

# ensure Dask workers don't use local storage
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.90
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95

# run Python script
python scripts/hyperparameters_search_slurm.py
