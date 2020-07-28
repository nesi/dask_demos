# # Hyper-parameters tuning on HPC (basic)
#
# TODO intro
# TODO link to other notebook for advanced stuff

# +
import pprint

from scipy import stats as st
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# -

# Load Breast Cancer Wisconsin dataset.

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Fit a simple multi-layer perceptron neural net.

mlp = make_pipeline(RobustScaler(), MLPClassifier()).fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, y_pred)
print(f"Baseline MLP accuracy is {mlp_acc * 100:.2f}%.")

# Tune hyper-parameters using a random search strategy.

params_dist = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (50, 50), (100,), (100, 100)],
    "mlpclassifier__alpha": st.loguniform(1e-6, 1e-2),
    "mlpclassifier__learning_rate_init": st.loguniform(5e-4, 5e-2),
    "mlpclassifier__early_stopping": [True, False],
}
mlp_tuned = RandomizedSearchCV(mlp, params_dist, random_state=42, n_iter=10)
mlp_tuned.fit(X_train, y_train)

y_pred_tuned = mlp_tuned.predict(X_test)
mlp_tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"Tuned MLP accuracy is {mlp_tuned_acc * 100:.2f}%.")
print(f"Best parameters found:")
pprint.pp(mlp_tuned.best_params_)

# Start a Dask cluster (see notes in README.md about additional configuration files).

cluster = SLURMCluster(cores=4, processes=1, memory="8GiB", walltime="0-00:30")
cluster.adapt(minimum_jobs=4, maximum_jobs=20)
client = Client(cluster)

# Scikit-learn uses [Joblib](https://joblib.readthedocs.io) to parallelize
# computations of many operations, including the randomized search on hyper-parameters.
# If we configure Joblib to use Dask as a backend, computations will be automatically
# scheduled and distributed on nodes of the HPC.

mlp_dask = RandomizedSearchCV(mlp, params_dist, random_state=42, n_iter=500)
with joblib.parallel_backend("dask", wait_for_workers_timeout=600):
    mlp_dask.fit(X_train, y_train)

# Enjoy an optimized model :).

y_pred_dask = mlp_dask.predict(X_test)
mlp_dask_acc = accuracy_score(y_test, y_pred_dask)
print(f"Tuned (w/ Dask) MLP accuracy is {mlp_dask_acc * 100:.2f}%.")
print(f"Best parameters found:")
pprint.pp(mlp_dask.best_params_)
