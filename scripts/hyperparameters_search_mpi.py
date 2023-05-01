import time
from urllib.parse import urlparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import joblib
import dask_mpi as dm
from dask.distributed import Client

if __name__ == "__main__":
    dm.initialize(local_directory="dask")

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

    client = Client()
    url = urlparse(client.dashboard_link)
    print(
        "### dashboard link: https://jupyter.nesi.org.nz/user-redirect/proxy/"
        f"{url.hostname}:{url.port}{url.path} ###",
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
