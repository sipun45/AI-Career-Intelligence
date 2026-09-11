import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import numpy as np
import mlflow

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# Configuration
# ==========================================

DATA_PATH = "data/raw/jobs.csv"
MODEL_PATH = "models/salary_model.pkl"

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Career Salary Monitoring"


# ==========================================
# Load Data
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["education_encoded"] = df["education"].map({
        "Bachelor": 0,
        "Master": 1
    })

    skill_columns = [
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud"
    ]

    df["total_skills"] = df[skill_columns].sum(axis=1)

    features = [
        "experience",
        "education_encoded",
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud",
        "total_skills"
    ]

    X = df[features]
    y = df["salary"]

    return X, y


# ==========================================
# Main Monitoring
# ==========================================

if __name__ == "__main__":

    print("===================================")
    print("MLFLOW MODEL MONITORING")
    print("===================================")

    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load model
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # Load data
    X, y = load_data()

    # Predictions
    predictions = model.predict(X)

    # Metrics
    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    # Start MLflow run
    with mlflow.start_run():

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.log_param(
            "model",
            "Linear Regression"
        )

        mlflow.log_param(
            "dataset",
            "jobs.csv"
        )

        mlflow.log_param(
            "monitoring_type",
            "production_performance"
        )

        # Save monitoring report
        report = pd.DataFrame({
            "actual_salary": y,
            "predicted_salary": predictions,
            "error": y - predictions
        })

        report.to_csv(
            "monitoring_report.csv",
            index=False
        )

        mlflow.log_artifact(
            "monitoring_report.csv"
        )

        print("\nMetrics recorded in MLflow.")

        print("\n===================================")
        print("MONITORING RESULTS")
        print("===================================")

        print(f"MAE  : ${mae:,.2f}")
        print(f"RMSE : ${rmse:,.2f}")
        print(f"R²   : {r2:.4f}")

        if r2 >= 0.90:
            print("\nStatus: GOOD")
        elif r2 >= 0.70:
            print("\nStatus: ACCEPTABLE")
        else:
            print("\nStatus: PERFORMANCE DEGRADATION")

    print("\nMLflow monitoring completed.")