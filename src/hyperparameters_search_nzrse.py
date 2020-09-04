# # Hyper-parameters tuning on HPC (basic)
#
# This demo illustrates one simple way to adapt a grid search strategy for
# hyper-parameters tuning to use HPC for the many parallel computations involved.
#
# In this example, we will rely on [Dask](https://dask.org) to do the heavy lifting,
# distributing the parallel operations on SLURM jobs. We'll see how it can be used
# as a backend for [Scikit-Learn](https://scikit-learn.org) estimators, with very
# little changes compared to a vanilla grid search.

# +
import time

import numpy as np
import scipy.stats as st
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

import joblib
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask_ml.model_selection import HyperbandSearchCV

import torch
from skorch import NeuralNetClassifier
from src.torch_models import SimpleMLP

# -

# Load MNIST data from [OpenML](https://www.openml.org/d/554).

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, train_size=5000, test_size=10000, random_state=42
)

# Fit a simple multi-layer perceptron neural net.

start = time.perf_counter()
mlp = MLPClassifier(random_state=42).fit(X_train, y_train)
elapsed = time.perf_counter() - start
print(f"Model fitting took {elapsed:0.2f}s.")

y_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred)
print(f"Baseline MLP test accuracy is {mlp_acc * 100:.2f}%.")

# Tune hyper-parameters using a random search strategy.

param_space = {
    "hidden_layer_sizes": st.randint(50, 200),
    "alpha": st.loguniform(1e-5, 1e-2),
    "learning_rate_init": st.loguniform(1e-4, 1e-1),
}
mlp_tuned = RandomizedSearchCV(
    MLPClassifier(random_state=42), param_space, random_state=42, verbose=1
)

# Start a Dask cluster using SLURM jobs as workers.
#
# There are a couple of things we need to configure here:
#
# - disabling the mechanism to write on disk when workers run out of memory,
# - memory, CPUs, maximum time and number of workers per SLURM job,
# - dask folders for log files and workers data.
#
# We recommend putting the log folder and workers data folders in your
# `/nesi/nobackup/<project_code>` folder, most indicated for temporary files
# (see [NeSI File Systems and Quotas](https://support.nesi.org.nz/hc/en-gb/articles/360000177256-NeSI-File-Systems-and-Quotas)).
#
# All of these options can be set in configuration files, see [Dask configuration](https://docs.dask.org/en/latest/configuration.html)
# and [Dask jobqueue configuration](https://jobqueue.dask.org/en/latest/configuration-setup.html)
# for more information.

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
client = Client(cluster)

# Spawn 20 workers and connect a client to be able use them.

cluster.scale(n=20)

# Scikit-learn uses [Joblib](https://joblib.readthedocs.io) to parallelize
# computations of many operations, including the randomized search on hyper-parameters.
# If we configure Joblib to use Dask as a backend, computations will be automatically
# scheduled and distributed on nodes of the HPC.

with joblib.parallel_backend(
    "dask", wait_for_workers_timeout=600, scatter=[X_train, y_train]
):
    start = time.perf_counter()
    mlp_tuned.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

n_jobs = len(mlp_tuned.cv_results_["params"]) * mlp_tuned.n_splits_
print(
    f"Model fitting took {elapsed:0.2f}s (equivalent to {elapsed / n_jobs:0.2f}s "
    "per model fit on a single node)."
)

# Enjoy an optimized model :).

y_pred_tuned = mlp_tuned.predict(X_test)
mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned MLP test accuracy is {mlp_tuned_acc * 100:.2f}%.")

print(f"Best hyper-parameters: {mlp_tuned.best_params_}")

# TODO Dask ML - hyperband
mlp_hyper = HyperbandSearchCV(
    MLPClassifier(random_state=42),
    param_space,
    max_iter=200,
    aggressiveness=4,
    random_state=42,
)

start = time.perf_counter()
mlp_hyper.fit(X_train, y_train, classes=np.unique(y))
elapsed = time.perf_counter() - start
print(f"Model fitting took {elapsed:0.2f}s.")

# TODO add timing vs. number of models

y_pred_hyper = mlp_hyper.predict(X_test)
mlp_hyper_acc = accuracy_score(y_test, y_pred_hyper)
print(f"MLP (hyperband) test accuracy is {mlp_hyper_acc * 100:.2f}%.")

# TODO GPU cluster

cluster.close()
client.close()

cluster = SLURMCluster(
    cores=4,
    processes=1,
    memory="4GiB",
    walltime="0-00:30",
    log_directory="../dask/logs",  # folder for SLURM logs for each worker
    local_directory="../dask",  # folder for workers data
    job_extra=["--gres gpu:1"],
    queue="gpu",
)
client = Client(cluster)

cluster.adapt(minimum=1, maximum=4)

# TODO Skorch

torch.manual_seed(0)
mlp_torch = NeuralNetClassifier(
    module=SimpleMLP,
    module__input_dim=X.shape[1],
    module__output_dim=len(np.unique(y)),
    optimizer=torch.optim.Adam,
    device="cuda",
)

param_space = {
    "module__hidden_dim": st.randint(50, 200),
    "module__dropout": st.uniform(),
    "optimizer__lr": st.loguniform(1e-4, 1e-1),
}
mlp_torch_hyper = HyperbandSearchCV(
    mlp_torch, param_space, max_iter=200, aggressiveness=4, random_state=42
)

start = time.perf_counter()
mlp_torch_hyper.fit(X_train.astype(np.float32), y_train)
elapsed = time.perf_counter() - start
print(f"Model fitting took {elapsed:0.2f}s.")

y_pred_torch = mlp_torch_hyper.predict(X_test.astype(np.float32))
mlp_torch_acc = accuracy_score(y_test, y_pred_torch)
print(f"MLP (PyTorch) test accuracy is {mlp_torch_acc * 100:.2f}%.")

# TODO nevergrad?
