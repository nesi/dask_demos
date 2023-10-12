import os
import time
import socket

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


if __name__ == "__main__":
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, parser="pandas", as_frame=False
    )
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=5000, test_size=10000, random_state=42
    )

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (200,)],
        "alpha": np.logspace(-5, -3, 3),
        "learning_rate_init": np.logspace(-4, -2, 3),
    }
    mlp_tuned = GridSearchCV(MLPClassifier(), param_grid, verbose=1)

    # fetch Slurm account from the Slurm job running this script
    account = os.environ["SLURM_JOB_ACCOUNT"]

    # Note: on NeSI, if `local_directory=None`, `/dev/shm/jobs/...` tmpfs folders
    # will be used. This should not be an issue (on the contrary, workers start
    # faster) as long as the workers don't spill memory to disk, which has been
    # disabled (see Dask configuration in the job submission script).

    cluster = SLURMCluster(
        memory="1GB",  # memory for Slurm job, not per worker
        cores=8,  # cores for Slurm job, not per worker
        processes=2,  # number of Dask worker per Slurm job
        walltime="0-00:20:00",  # make it long enough to outlive main process
        account=account,
        interface="ib0",  # ensure infiniband interface is used
        queue="milan",  # not needed unless targeting a specific partition
        log_directory="dask/logs",  # worker Slurm jobs logs folder
        local_directory=None,  # see comment above ;-)
        worker_extra_args=["--memory-limit 1GB"],  # danger zone: oversubscribe memory
        # TODO show how to load a module?
    )
    with cluster, Client(cluster) as client:
        cluster.scale(jobs=10)
        client = Client(cluster)

        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()["services"]["dashboard"]
        print(
            "### dashboard link: https://jupyter.nesi.org.nz/user-redirect/proxy/"
            f"{host}.ib.hpcf.nesi.org.nz:{port}/status ###",
            flush=True,
        )

        with joblib.parallel_backend("dask", wait_for_workers_timeout=600):
            start = time.perf_counter()
            mlp_tuned.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

        n_jobs = len(mlp_tuned.cv_results_["params"]) * mlp_tuned.n_splits_
        print(
            f"Model fitting took {elapsed:0.2f}s "
            f"(equivalent to {elapsed / n_jobs:0.2f}s per model fit on a single node)."
        )

        y_pred_tuned = mlp_tuned.predict(X_test)
        mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
        print(f"Tuned MLP test accuracy is {mlp_tuned_acc * 100:.2f}%.")

    print(f"Best hyper-parameters: {mlp_tuned.best_params_}")
