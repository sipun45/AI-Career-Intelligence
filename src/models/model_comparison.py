import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/raw/jobs.csv")


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

X = df[features].values
y = df["salary"].values


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
# 4. Train Traditional ML Models
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

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


results = []


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
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

    results.append({
        "Model": name,
        "MAE": mae,
        "R2": r2
    })


# ==========================================
# 5. PyTorch Neural Network
# ==========================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()


X_train_scaled = X_scaler.fit_transform(
    X_train
)

X_test_scaled = X_scaler.transform(
    X_test
)


y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
)


class SalaryNeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


model = SalaryNeuralNetwork()


X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)


loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 6. Train Neural Network
# ==========================================

for epoch in range(300):

    model.train()

    predictions = model(
        X_train_tensor
    )

    loss = loss_function(
        predictions,
        y_train_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


# ==========================================
# 7. Neural Network Prediction
# ==========================================

model.eval()

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).numpy()


predictions = y_scaler.inverse_transform(
    predictions_scaled
).flatten()


mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


results.append({
    "Model": "PyTorch Neural Network",
    "MAE": mae,
    "R2": r2
})


# ==========================================
# 8. Create Results DataFrame
# ==========================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "MAE"
)


# ==========================================
# 9. Display Results
# ==========================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================\n")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 10. Best Model
# ==========================================

best_model = results_df.iloc[0]


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(
    f"Model: {best_model['Model']}"
)

print(
    f"MAE: ${best_model['MAE']:.2f}"
)

print(
    f"R2 Score: {best_model['R2']:.4f}"
)