# # Hyper-parameters tuning on HPC (basic)
#
# This demo illustrates one simple way to adapt a grid search strategy for
# hyper-parameters tuning to use HPC for the many parallel computations involved.
#
# In this example, I will rely on [Dask](https://dask.org) to do the heavy lifting,
# distributing the parallel operations on SLURM jobs. We'll see how it can be used
# as a backend for [Scikit-Learn](https://scikit-learn.org) estimators, with very
# little changes compared to a vanilla grid search.
#
# TODO link to other notebook for advanced stuff

# +
import pprint
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# -

# Load MNIST data from [OpenML](https://www.openml.org/d/554).

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, train_size=10000, test_size=10000, random_state=42
)

# Fit a simple multi-layer perceptron neural net.

start = time.perf_counter()
mlp = MLPClassifier().fit(X_train, y_train)
elapsed = time.perf_counter() - start
print(f"Model fitting took {elapsed:0.2f}s.")

y_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred)
print(f"Baseline MLP test accuracy is {mlp_acc * 100:.2f}%.")

# Tune hyper-parameters using a random search strategy.

param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (200,)],
    "alpha": np.logspace(-5, -3, 3),
    "learning_rate_init": np.logspace(-4, -2, 3),
}
mlp_tuned = GridSearchCV(MLPClassifier(), param_grid, verbose=1)

# Start a Dask cluster (see notes in README.md about additional configuration files).

cluster = SLURMCluster(cores=10, processes=2, memory="8GiB", walltime="0-00:30")
cluster.scale(n=10)
client = Client(cluster)

# Scikit-learn uses [Joblib](https://joblib.readthedocs.io) to parallelize
# computations of many operations, including the randomized search on hyper-parameters.
# If we configure Joblib to use Dask as a backend, computations will be automatically
# scheduled and distributed on nodes of the HPC.

with joblib.parallel_backend("dask", wait_for_workers_timeout=600):
    start = time.perf_counter()
    mlp_tuned.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

n_jobs = len(mlp_tuned.cv_results_["params"]) * mlp_tuned.n_splits_
print(f"Model fitting took {elapsed:0.2f}s ({elapsed / n_jobs:0.2f}s per model fit).")

# Enjoy an optimized model :).

y_pred_tuned = mlp_tuned.predict(X_test)
mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned MLP test accuracy is {mlp_tuned_acc * 100:.2f}%.")

print(f"Best hyper-parameters: {mlp_tuned.best_params_}")

# TODO notes about what could go wrong (memory consumption)
