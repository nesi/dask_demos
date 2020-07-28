# HPC for Data Science demos

TODO intro


## Installation

- TODO venv recommendation + pip install
```
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

python3 -m ipykernel install --user --name hpc_for_datascience_demos
```

- TODO labextensions (dask, proxy-server, pyviz)

- TODO dask configuration (log folder, worker folder)


## Demos

- TODO hyperparameters search (intro)
- TODO hyperparameters search (advanced)
- TODO interactive visualization (intro)
- TODO interactive visualization (advanced)
- TODO deep-learning multi-gpu training
- TODO Approximate Bayesian Computation


## Maintainer's notes

The demos are written as plain scripts, converted into notebooks using
[jupytext](https://github.com/mwouts/jupytext).
- TODO convertion command
- TODO hooks for flake8 and black
- TODO pin dependencies
