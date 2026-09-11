import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# MLflow Configuration
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()


# ==========================================
# Experiment
# ==========================================

experiment = client.get_experiment_by_name(
    "Career Salary Prediction"
)


if experiment is None:

    print("Experiment not found.")

    raise SystemExit


experiment_id = experiment.experiment_id


# ==========================================
# Find Successful Runs
# ==========================================

runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=[
        "metrics.MAE ASC"
    ]
)


if not runs:

    print("No MLflow runs found.")

    raise SystemExit


# ==========================================
# Display Runs
# ==========================================

print("\n==========================================")
print("MLFLOW RUNS")
print("==========================================\n")


for run in runs:

    model_name = run.data.params.get(
        "model",
        "Unknown"
    )

    mae = run.data.metrics.get(
        "MAE",
        None
    )

    r2 = run.data.metrics.get(
        "R2",
        None
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Run ID: {run.info.run_id}"
    )

    if mae is not None:
        print(
            f"MAE: ${mae:.2f}"
        )

    if r2 is not None:
        print(
            f"R2: {r2:.4f}"
        )

    print("------------------------------------------")


# ==========================================
# Select Best Run
# ==========================================

best_run = runs[0]

best_run_id = best_run.info.run_id

best_model_name = best_run.data.params.get(
    "model",
    "Unknown"
)


print("\n==========================================")
print("BEST MODEL")
print("==========================================")


print(
    f"Model: {best_model_name}"
)

print(
    f"Run ID: {best_run_id}"
)


# ==========================================
# Model URI
# ==========================================

model_uri = (
    f"runs:/{best_run_id}/"
    f"linear_regression_model"
)


# ==========================================
# Register Model
# ==========================================

model_name = "CareerSalaryPredictor"


try:

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print("\n==========================================")
    print("MODEL REGISTERED")
    print("==========================================")

    print(
        f"Name: {result.name}"
    )

    print(
        f"Version: {result.version}"
    )

except Exception as error:

    print("\nModel registration failed:")

    print(error)