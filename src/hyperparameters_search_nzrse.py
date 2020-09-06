# # Hyper-parameters tuning on HPC - NZ RSE 2020
#
# This demo illustrates one simple way multiple ways to adapt a random search
# strategy for hyper-parameters tuning to use HPC for the many parallel
# computations involved.
#
# In this example, we will rely on [Dask](https://dask.org) to do the heavy lifting,
# distributing the parallel operations on SLURM jobs.

# +
import json
import warnings

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
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
import skorch
from src.torch_models import SimpleCNN

# -

# Load MNIST data from [OpenML](https://www.openml.org/d/554).

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0
y = y.astype(int)

# This dataset contains images of digits. Here is a sample.

_, axes = plt.subplots(1, 10, figsize=(12, 5))
for ax, digit in zip(axes, X):
    ax.imshow(digit.reshape(28, 28))
    ax.axis("off")

# To keep this example code quick, let's use only a subset of the whole data set
# as train and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, train_size=5000, test_size=10000, random_state=42
)

# Fit a simple multi-layer perceptron neural net.

# %%time
mlp = MLPClassifier(random_state=42).fit(X_train, y_train)

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
    MLPClassifier(random_state=42), param_space, n_iter=10, random_state=42, verbose=1
)

# Start a Dask cluster using SLURM jobs as workers.
#
# There are a couple of things we need to configure here:
#
# - disabling the mechanism to write on disk when workers run out of memory,
# - memory, CPUs, maximum time and number of workers per SLURM job,
# - dask folders for log files and workers data.
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
    queue="bigmem",
)
client = Client(cluster)

# Spawn 20 workers and connect a client to be able use them.

cluster.scale(n=20)
client.wait_for_workers(1)

# Scikit-learn uses [Joblib](https://joblib.readthedocs.io) to parallelize
# computations of many operations, including the randomized search on
# hyper-parameters. If we configure Joblib to use Dask as a backend,
# computations will be automatically scheduled and distributed on nodes of the
# HPC platform.

# %%time
with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
    mlp_tuned.fit(X_train, y_train)

# Enjoy an optimized model :).

y_pred_tuned = mlp_tuned.predict(X_test)
mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned MLP test accuracy is {mlp_tuned_acc * 100:.2f}%.")

print(f"Best hyper-parameters:\n{json.dumps(mlp_tuned.best_params_, indent=4)}")

# This first approach requires very little changes but is far from optimal from
# a ressource usage perspective. The [Dask-ML](https://ml.dask.org/) package
# implements a more advanced algorithm, [Hyperband](https://arxiv.org/abs/1603.06560),
# designed to better use a finite bugdet, using early stopping of bad
# configurations. It relies on Scikit-learn API, assuming the estimator
# implements the `partial_fit` method.

mlp_hyper = HyperbandSearchCV(
    MLPClassifier(random_state=42),
    param_space,
    max_iter=200,
    aggressiveness=4,
    random_state=42,
)

# %%time
_ = mlp_hyper.fit(X_train, y_train, classes=np.unique(y))

y_pred_hyper = mlp_hyper.predict(X_test)
mlp_hyper_acc = accuracy_score(y_test, y_pred_hyper)
print(f"MLP (Hyperband) test accuracy is {mlp_hyper_acc * 100:.2f}%.")

print(f"Best hyper-parameters:\n{json.dumps(mlp_hyper.best_params_, indent=4)}")

# Now if we want to try some deep learning models trained with GPUs, we need to
# start a new Dask cluster, requesting the right resources. First let's stop the
# current cluster.

cluster.close()
client.close()

# Then start a new cluster, explicitly asking for GPUs.

cluster = SLURMCluster(
    cores=4,
    processes=1,
    memory="4GiB",
    walltime="0-00:30",
    log_directory="../dask/logs",
    local_directory="../dask",
    job_extra=["--gres gpu:1"],  # passed to job script to request a GPU
    queue="gpu",  # use the GPU partition
)
client = Client(cluster)

# Here we use an adaptive scaling strategy, asking Dask scheduler to start at
# least one worker and letting it spawning on the fly more if needed.

cluster.adapt(minimum=1, maximum=4)
client.wait_for_workers(1)

# For the deep learning part, let's use a simple convolutional neural network
# implemented using [Pytorch](https://pytorch.org/). We make use of [Skorch](https://github.com/skorch-dev/skorch)
# to make it in a Scikit-learn compatible estimator.


def build_model(device):
    torch.manual_seed(0)
    return skorch.NeuralNetClassifier(
        module=SimpleCNN,
        module__input_dims=(28, 28),
        module__output_dim=len(np.unique(y)),
        module__n_chans=32,
        module__hidden_dim=100,
        module__dropout=0.5,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        device=device,
    )


# Before training the model, we will compare timing on CPU and GPU. This will
# also illustrate another aspect of the Dask API: direct submission of tasks
# on the cluster.
#
# We first dispatch training data on the cluster using `client.scatter`, to save
# some bandwidth later. Here `X_train_future` is a **future** object, no yet
# computed.

X_train_future = client.scatter(X_train.astype(np.float32))
X_train_future

# First, let's instantiate the CPU version of the model. `client.submit` send
# the computation, here `model.fit`, on a worker of the cluster and returns a
# future object. This is a non-blocking call. To wait and get the result, we can
# use the `result` method of the future object.

mlp_torch = build_model("cpu")

# %%time
mlp_torch_future = client.submit(mlp_torch.fit, X_train_future, y_train)
mlp_torch = mlp_torch_future.result()

# And now let's try the GPU version.

mlp_torch = build_model("cuda")

# %%time
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp_torch_future = client.submit(mlp_torch.fit, X_train_future, y_train)
    mlp_torch = mlp_torch_future.result()

# Thanks to Skorch, our model is compatible with Scikit-learn API, and thus can
# be used with Dask-ML meta-estimators. Hence we will use Hyberband to search
# for the best hyper-parameters.

param_space = {
    "module__n_chans": st.randint(10, 64),
    "module__hidden_dim": st.randint(50, 200),
    "module__dropout": st.uniform(),
    "optimizer__lr": st.loguniform(1e-4, 1e-1),
}
mlp_torch = HyperbandSearchCV(
    build_model("cuda"), param_space, max_iter=20, random_state=42
)

# %%time
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp_torch.fit(X_train.astype(np.float32), y_train)

y_pred_torch = mlp_torch.predict(X_test.astype(np.float32))
mlp_torch_acc = accuracy_score(y_test, y_pred_torch)
print(f"CNN (PyTorch & Hyperband) test accuracy is {mlp_torch_acc * 100:.2f}%.")

print(f"Best hyper-parameters:\n{json.dumps(mlp_torch.best_params_, indent=4)}")
