import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# Paths
# ==========================================

DATA_PATH = "data/raw/jobs.csv"
MODEL_PATH = "models/salary_model.pkl"

# ==========================================
# Load Model
# ==========================================

def load_model():

    model = joblib.load(MODEL_PATH)

    print("Linear Regression model loaded successfully.")

    return model


# ==========================================
# Load Data
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ==========================================
# Prepare Features
# ==========================================

def prepare_features(df):

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
# Monitor Model
# ==========================================

def monitor_model(model, X, y):

    predictions = model.predict(X)

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

    print("\n===================================")
    print("REAL MODEL MONITORING")
    print("===================================")

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    print("\n===================================")
    print("PREDICTIONS")
    print("===================================")

    for actual, predicted in zip(
        y.values,
        predictions
    ):

        print(
            f"Actual: ${actual:,.0f} "
            f"| Predicted: ${predicted:,.0f}"
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================
# Performance Check
# ==========================================

def performance_check(metrics):

    print("\n===================================")
    print("MODEL STATUS")
    print("===================================")

    if metrics["R2"] >= 0.90:

        print("Status: GOOD")

    elif metrics["R2"] >= 0.70:

        print("Status: ACCEPTABLE")

    else:

        print("Status: PERFORMANCE DEGRADATION")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    model = load_model()

    df = load_data()

    X, y = prepare_features(df)

    metrics = monitor_model(
        model,
        X,
        y
    )

    performance_check(metrics)