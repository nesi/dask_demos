#!/usr/bin/env bash
#SBATCH --account=nesi99999
#SBATCH --time=00-00:10:00
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out
#SBATCH --ntasks=2 --mem-per-cpu=1G --cpus-per-task=1
#SBATCH hetjob
#SBATCH --ntasks=20 --mem-per-cpu=1G --cpus-per-task=4

# load environment modules and activate conda environment
module purge && module load Miniconda3/22.11.1-1 impi/2021.5.1-GCC-11.3.0
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

conda deactivate
conda activate ./venv

# ensure Dask workers don't use local storage
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95

# run Python script
srun --het-group=0-1 python scripts/hyperparameters_search_mpi.py
