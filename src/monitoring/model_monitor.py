import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================
# Configuration
# ==============================

DATA_PATH = "data/raw/jobs.csv"


# ==============================
# Load Data
# ==============================

def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ==============================
# Model Evaluation
# ==============================

def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics


# ==============================
# Prediction Monitoring
# ==============================

def monitor_predictions(y_true, y_pred):

    errors = np.abs(
        np.array(y_true) - np.array(y_pred)
    )

    print("\n===================================")
    print("PREDICTION MONITORING")
    print("===================================")

    print(f"Average Error : ${errors.mean():,.2f}")
    print(f"Maximum Error : ${errors.max():,.2f}")
    print(f"Minimum Error : ${errors.min():,.2f}")

    metrics = evaluate_model(
        y_true,
        y_pred
    )

    print("\nMODEL METRICS")
    print("-----------------------------------")

    print(f"MAE  : ${metrics['MAE']:,.2f}")
    print(f"RMSE : ${metrics['RMSE']:,.2f}")
    print(f"R²   : {metrics['R2']:.4f}")

    return metrics


# ==============================
# Simple Performance Check
# ==============================

def performance_check(metrics):

    print("\n===================================")
    print("MODEL PERFORMANCE CHECK")
    print("===================================")

    if metrics["R2"] >= 0.90:

        print("Status: GOOD")

    elif metrics["R2"] >= 0.70:

        print("Status: ACCEPTABLE")

    else:

        print("Status: PERFORMANCE DEGRADATION")


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    df = load_data()

    # Example monitoring predictions
    # Replace these with real model predictions
    y_true = df["salary"].values

    # Example predictions
    # Small variation added only for demonstration
    np.random.seed(42)

    y_pred = (
        y_true +
        np.random.normal(
            0,
            3000,
            len(y_true)
        )
    )

    metrics = monitor_predictions(
        y_true,
        y_pred
    )

    performance_check(metrics)