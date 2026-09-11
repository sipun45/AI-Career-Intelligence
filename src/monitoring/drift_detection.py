import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np


DATA_PATH = "data/raw/jobs.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print(f"Rows: {len(df)}")

    return df


# ==========================================
# Detect Numerical Drift
# ==========================================

def detect_drift(reference, current, threshold=0.20):

    reference_mean = np.mean(reference)
    current_mean = np.mean(current)

    if reference_mean == 0:
        return 0

    drift_score = abs(
        current_mean - reference_mean
    ) / abs(reference_mean)

    return drift_score


# ==========================================
# Analyze Features
# ==========================================

def analyze_drift(df):

    features = [
        "experience",
        "python",
        "java",
        "sql",
        "ml",
        "deep_learning",
        "cloud"
    ]

    print("\n===================================")
    print("DATA DRIFT DETECTION")
    print("===================================")

    for feature in features:

        reference = df[feature].values

        # Simulated new/current data
        # Used only to demonstrate monitoring
        np.random.seed(42)

        current = reference + np.random.normal(
            0,
            0.1,
            len(reference)
        )

        score = detect_drift(
            reference,
            current
        )

        print(f"\nFeature: {feature}")
        print(f"Drift Score: {score:.4f}")

        if score > 0.20:
            print("Status: DRIFT DETECTED")
        else:
            print("Status: STABLE")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = load_data()

    analyze_drift(df)