# HPC for Data Science - Demos

This repository contains various notebooks to demonstrate how to run data science
related tasks on NeSI's HPC platform.


## Installation

First, to be able to run the notebooks, you will need to have access to
JupyterLab on the HPC plateform, either using the dedicated [JupyterLab module](https://support.nesi.org.nz/hc/en-gb/articles/360001093315-JupyterLab)
or logging in via JupyterHub (TODO link to guide).

Once you are connected to a JupyterLab instance, open a new terminal, via the
Launcher or the File menu.

Then clone this repository:
```
git clone https://github.com/nesi/hpc_for_datascience_demos
```

And install all dependencies, either in your home space:
```
cd hpc_for_datascience_demos
pip install --user requirements.txt
```

Or create a virtual environment to cleanly isolate this project from others:
```
cd hpc_for_datascience_demos
make venv
```
This command will create a virtual environment in the `venv` folder and register
it as a new kernel named `hpc_for_datascience_demos` to be used with the notebooks.

- TODO labextensions (dask, proxy-server, pyviz), should be already installed
- TODO dask configuration (log folder, worker folder)


## Demos

- [hyperparameters search (basic)](notebooks/hyperparameters_search_basic.ipynb)
- TODO hyperparameters search (advanced)
- TODO interactive visualization (intro)
- TODO interactive visualization (advanced)
- TODO deep-learning multi-gpu training
- TODO Approximate Bayesian Computation


## Maintainer's notes

Demos are written as plain scripts, converted into notebooks using [jupytext](https://github.com/mwouts/jupytext).

A [Makefile](Makefile) is provided to automate execution and conversion of the
scripts into notebooks and static html documents, just run `make`.

Dependencies are pinned in the [requirements-pinned.txt](requirements-pinned.txt)
file, to keep a trace of the execution environment when generating the notebooks.

- TODO pre-commit hooks using flake8 and black