# Dask Demos

This repository contains to demos to show how to run [Dask](https://www.dask.org/) on NeSI's HPC platform.


## Installation

These demos are meant to be run via [Jupyter on NeSI](https://support.nesi.org.nz/hc/en-gb/articles/360001555615-Jupyter-on-NeSI).
First, make sure to first log in https://jupyter.nesi.org.nz.

Once you are connected to a JupyterLab instance, open a new terminal, via the
Launcher or the File menu.

Then clone this repository:

```
git clone https://github.com/nesi/dask_demos
```

And install all dependencies in a conda environment to cleanly isolate this project from others:

```
cd hpc_for_datascience_demos
module purge && module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda env create --solver=libmamba -f environment.lock.yml -p ./venv
```

*Note: The `environment.lock.yml` file has been generated from a conda environment created with the `environment.yml` file and then exported with*

```
conda env export -p ./venv --no-builds | sed '/^name: .*/d; /^prefix: .*/d' > environment.lock.yml
```

The last command will create a conda environment in the `venv` folder.
Register it as a new jupyter kernel named `dask_demos` and ensure that
it uses the right [environment modules](https://support.nesi.org.nz/hc/en-gb/articles/360001113076-The-HPC-environment-) using the `nesi-add-kernel` tool (more information in our [dedicated support page](https://support.nesi.org.nz/hc/en-gb/articles/4414958674831-Jupyter-kernels-Tool-assisted-management)).

```
module purge && module load JupyterLab
nesi-add-kernel -p ./venv -- dask_demos CUDA/11.6.2
```

*Note: You can later remove this jupyter kernel using*

```
module purge && module load JupyterLab
jupyter-kernelspec remove dask_demos
```


## Demos

Make sure to select the `dask_demos` kernel when running any of the provided notebooks.

- The [dask basics](notebooks/dask_basics.ipynb) notebook is used to introduce Dask.
- The [hyperparameters search](notebooks/hyperparameters_search.ipynb) notebook shows how to adapt a Scikit-Learn grid search to run in parallel on HPC.
- The [hyperparameters search](scripts/hyperparameters_search.py) script (and corresponding [Slurm script](scripts/hyperparameters_search.sl)) illustrates how to deploy the grid search example in batch job using the `LocalCluster` backend.
- The [hyperparameters search (MPI)](scripts/hyperparameters_search_mpi.py) script (and corresponding [Slurm script](scripts/hyperparameters_search_mpi.sl)) presents the grid search example in batch job using the Dask-MPI backend.
- The [hyperparameters search (NZ RSE 2020)](notebooks/hyperparameters_search_nzrse.ipynb) notebook corresponds to the demo presented at [NZ RSE conference 2020](https://www.rseconference.nz/).
