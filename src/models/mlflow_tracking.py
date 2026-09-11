import os

# ==========================================
# Prevent excessive CPU / memory usage
# ==========================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ==========================================
# Imports
# ==========================================

import pandas as pd

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv(
    "data/raw/jobs.csv"
)


# ==========================================
# 2. Feature Engineering
# ==========================================

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


df["total_skills"] = df[
    skill_columns
].sum(axis=1)


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


# ==========================================
# 3. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. MLflow Experiment
# ==========================================

mlflow.set_experiment(
    "Career Salary Prediction"
)


# ==========================================
# 5. Define Models
# ==========================================

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)


# ==========================================
# 6. Linear Regression
# ==========================================

print("\n==========================================")
print("Training: Linear Regression")
print("==========================================")


with mlflow.start_run(
    run_name="Linear Regression"
):

    linear_model.fit(
        X_train,
        y_train
    )


    predictions = linear_model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    print(
        f"MAE: ${mae:.2f}"
    )

    print(
        f"R2 Score: {r2:.4f}"
    )


    mlflow.log_param(
        "model",
        "Linear Regression"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )


    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "R2",
        r2
    )


    mlflow.sklearn.log_model(
        linear_model,
        name="linear_regression_model"
    )


    print(
        "Experiment logged to MLflow."
    )


# ==========================================
# 7. Random Forest
# ==========================================

print("\n==========================================")
print("Training: Random Forest")
print("==========================================")


with mlflow.start_run(
    run_name="Random Forest"
):

    random_forest_model.fit(
        X_train,
        y_train
    )


    predictions = random_forest_model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    print(
        f"MAE: ${mae:.2f}"
    )

    print(
        f"R2 Score: {r2:.4f}"
    )


    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )


    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "R2",
        r2
    )


    mlflow.sklearn.log_model(
        random_forest_model,
        name="random_forest_model"
    )


    print(
        "Experiment logged to MLflow."
    )


# ==========================================
# 8. XGBoost
# ==========================================

print("\n==========================================")
print("Training: XGBoost")
print("==========================================")


with mlflow.start_run(
    run_name="XGBoost"
):

    xgb_model.fit(
        X_train,
        y_train
    )


    predictions = xgb_model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    print(
        f"MAE: ${mae:.2f}"
    )

    print(
        f"R2 Score: {r2:.4f}"
    )


    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        3
    )

    mlflow.log_param(
        "learning_rate",
        0.05
    )

    mlflow.log_param(
        "random_state",
        42
    )


    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "R2",
        r2
    )


    # IMPORTANT:
    # Use XGBoost's own MLflow flavor.

    mlflow.xgboost.log_model(
        xgb_model,
        name="xgboost_model"
    )


    print(
        "Experiment logged to MLflow."
    )


# ==========================================
# 9. Finished
# ==========================================

print("\n==========================================")
print("MLflow Tracking Complete")
print("==========================================")


print(
    "\nOpen MLflow UI:"
)

print(
    "http://127.0.0.1:5000"
)