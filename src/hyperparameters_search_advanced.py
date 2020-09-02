# # Hyper-parameters tuning on HPC (advanced)
#
# TODO intro

# TODO goals:
# - advanced strategies (hyperband, Bayesian optimization)
# - long running jobs & partial_fit
# - large dataset & partial fit
# - non scikit-learn compatible API (ask/tell)
# - GPUs

# +
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import torch
from skorch import NeuralNetClassifier
from src.torch_models import SimpleMLP

import joblib
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.model_selection import GridSearchCV

# -

# TODO start dask cluster

dask.config.set(
    {
        "distributed.worker.memory.target": False,  # avoid spilling to disk
        "distributed.worker.memory.spill": False,  # avoid spilling to disk
    }
)
cluster = SLURMCluster(
    cores=10,
    processes=2,
    memory="8GiB",
    walltime="0-00:30",
    log_directory="../dask/logs",  # folder for SLURM logs for each worker
    local_directory="../dask",  # folder for workers data
)
cluster.scale(n=20)
client = Client(cluster)

# Note:
# For testing purpose, use a local Dask cluster to check everything works, for
# example running few iterations on a smaller dataset, as follows
# ```python
# from dask.distributed import Client
# client = Client(n_workers=1, processes=False)
# `

# TODO load mnist

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.astype(np.float32)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# TODO show some samples

# TODO skorch model (why? scikit-learn API, CPU/GPU, partial fit, mention keras wrapper/issues?)
# https://skorch.readthedocs.io/en/stable/user/quickstart.html
# https://github.com/skorch-dev/skorch/blob/master/notebooks/MNIST.ipynb
# https://github.com/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb
# TODO use log_softmax?

torch.manual_seed(0)
net = NeuralNetClassifier(
    module=SimpleMLP,
    module__input_dim=X.shape[1],
    module__output_dim=len(np.unique(y)),
    module__hidden_dim=50,
    module__dropout=0.5,
)
mlp = make_pipeline(MinMaxScaler(), net)
_ = mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred)
print(f"Simple MLP test accuracy is {mlp_acc * 100:.2f}%.")

# TODO dask-ml grid search (factorize preprocessing)
# https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html
# https://ml.dask.org/modules/generated/dask_ml.preprocessing.StandardScaler.html#dask_ml.preprocessing.StandardScaler
# data as dask-array & float32 (client.scatter), grid search -> standard scaler -> neural net
# visualize graph!
# note: could use incrementalCV if data too large

param_grid = {
    "neuralnetclassifier__module__hidden_dim": [50, 100, 200],
    "neuralnetclassifier__module__dropout": [0.2, 0.5, 0.8],
    # add learning rate (ADAM!?)
}
mlp.set_params(neuralnetclassifier__verbose=0)
mlp_tuned = GridSearchCV(mlp, param_grid)

start = time.perf_counter()
mlp_tuned.fit(X_train, y_train)
elapsed = time.perf_counter() - start

n_jobs = len(mlp_tuned.cv_results_["params"]) * mlp_tuned.n_splits_
print(
    f"Model fitting took {elapsed:0.2f}s (equivalent to {elapsed / n_jobs:0.2f}s "
    "per model fit on a single node)."
)

# TODO use dask array and lower memory on workers?
import dask.array as da

X_train = da.array(X_train).rechunk({0: 500})

# TODO dask-ml hyperband
# https://ml.dask.org/hyper-parameter-search.html


# TODO ask & tell interfaces (scikit-optimize, Ax service API, nevergrad)
# use nevergrad because parallel ask
# loop: ask, client.submit cross-validated on preprocess + model, tell


def score_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    return loss(y_test, yhat)


def cross_val_score(model, X, y, scorer):
    kf = KFold()
    scores = [
        score_model(model, X[train], y[train], X[test], y[test])
        for train, test in kf.split(X)
    ]
    return np.mean(scores)


# TODO scikit-optimize BayesSearchCV?
# TODO nevergrad executor concurrent?
# TODO switch to GPU training (change dask cluster and skorch?)
