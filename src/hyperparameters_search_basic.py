# # Hyper-parameters tuning on HPC (basic)
#
# TODO intro
# TODO link to other notebook for advanced stuff

from scipy import stats as st
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Load MNIST data from [OpenML](https://www.openml.org/d/554).

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=42
)

# Fit a simple multi-layer perceptron neural net.

mlp = MLPClassifier(hidden_layer_sizes=(10,)).fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred)
print(f"Baseline MLP accuracy is {mlp_acc * 100:.2f}%.")

# Tune hyper-parameters using a random search strategy.

params_dist = {
    "hidden_layer_sizes": [(10,), (50,), (20, 20)],
    "alpha": st.loguniform(1e-5, 1e-2),
    "batch_size": [100, 200, 300, 400],
    "learning_rate_init": st.loguniform(5e-4, 5e-2),
    "early_stopping": [True, False]
}
mlp_tuned = RandomizedSearchCV(
    MLPClassifier(), params_dist, random_state=42
)

# Start a Dask cluster (see notes in README.md about additional configuration files).

cluster = SLURMCluster(cores=4, processes=1, memory="8GiB", walltime="0-00:30")
cluster.adapt(minimum_jobs=4, maximum_jobs=20)
client = Client(cluster)

# Scikit-learn uses [Joblib](https://joblib.readthedocs.io) to parallelize
# computations of many operations, including the randomized search on hyper-parameters.
# If we configure Joblib to use Dask as a backend, computations will be automatically
# scheduled and distributed on nodes of the HPC.

# distributed the dataset on workers
#client.scatter(X_train, broadcast=True)

with joblib.parallel_backend("dask", wait_for_workers_timeout=600):
    mlp_tuned.fit(X_train, y_train)

# Enjoy an optimized model :).

y_pred_tuned = mlp_tuned.predict(X_test)
mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned MLP accuracy is {mlp_tuned_acc * 100:.2f}%.")
print(f"Best parameters found {mlp_tuned.best_params_}.")