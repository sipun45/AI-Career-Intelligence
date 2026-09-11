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
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


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
# 3. Define Models
# ==========================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
}


# ==========================================
# 4. 5-Fold Cross Validation
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ==========================================
# 5. Evaluate Each Model
# ==========================================

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        return_train_score=False
    )


    mae_scores = -scores["test_mae"]

    r2_scores = scores["test_r2"]


    results.append({

        "Model": name,

        "Average MAE":
            mae_scores.mean(),

        "MAE Std":
            mae_scores.std(),

        "Average R2":
            r2_scores.mean(),

        "R2 Std":
            r2_scores.std()
    })


# ==========================================
# 6. Results DataFrame
# ==========================================

results_df = pd.DataFrame(
    results
)


# Sort by MAE

results_df = results_df.sort_values(
    "Average MAE"
)


# ==========================================
# 7. Display Results
# ==========================================

print("\n==========================================")
print("5-FOLD CROSS VALIDATION")
print("==========================================\n")


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 8. Best Model
# ==========================================

best_model = results_df.iloc[0]


print("\n==========================================")
print("BEST MODEL")
print("==========================================")


print(
    f"Model: {best_model['Model']}"
)


print(
    f"Average MAE: ${best_model['Average MAE']:.2f}"
)


print(
    f"Average R2: {best_model['Average R2']:.4f}"
)


print(
    f"MAE Std: ${best_model['MAE Std']:.2f}"
)


print(
    f"R2 Std: {best_model['R2 Std']:.4f}"
)


# ==========================================
# 9. Fold-by-Fold Results
# ==========================================

print("\n==========================================")
print("INTERPRETATION")
print("==========================================")


print(
    "\n5-fold cross-validation evaluates "
    "each model on multiple train/test splits."
)


print(
    "\nLower MAE = better"
)


print(
    "Higher R2 = better"
)


print(
    "Lower Std = more stable performance"
)