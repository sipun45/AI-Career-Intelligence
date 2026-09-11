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
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


X = df[features].values

y = df["salary"].values.reshape(-1, 1)


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
# 4. Create Scalers
# ==========================================

X_scaler = StandardScaler()

y_scaler = StandardScaler()


# ==========================================
# 5. Fit Scalers ONLY on Training Data
# ==========================================

X_train = X_scaler.fit_transform(
    X_train
)

X_test = X_scaler.transform(
    X_test
)


y_train = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


# ==========================================
# 6. Save Scalers
# ==========================================

os.makedirs(
    "models/deep_learning",
    exist_ok=True
)


joblib.dump(
    X_scaler,
    "models/deep_learning/X_scaler.pkl"
)


joblib.dump(
    y_scaler,
    "models/deep_learning/y_scaler.pkl"
)


print("Scalers saved successfully.")


# ==========================================
# 7. Validation Split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 8. Convert Data to PyTorch Tensors
# ==========================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)


y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)


X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)


y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ==========================================
# 9. Define Neural Network
# ==========================================

class SalaryNeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input Layer
            nn.Linear(9, 32),

            # Activation
            nn.ReLU(),

            # Hidden Layer
            nn.Linear(32, 16),

            # Activation
            nn.ReLU(),

            # Output Layer
            nn.Linear(16, 1)
        )


    def forward(self, x):

        return self.network(x)


# ==========================================
# 10. Create Model
# ==========================================

model = SalaryNeuralNetwork()


# ==========================================
# 11. Loss Function
# ==========================================

loss_function = nn.MSELoss()


# ==========================================
# 12. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 13. Training Configuration
# ==========================================

epochs = 500

best_val_loss = float("inf")

patience = 30

patience_counter = 0


# ==========================================
# 14. Loss History
# ==========================================

train_losses = []

val_losses = []


# ==========================================
# 15. Training + Validation
# ==========================================

for epoch in range(epochs):


    # ======================================
    # Training
    # ======================================

    model.train()


    train_predictions = model(
        X_train_tensor
    )


    train_loss = loss_function(
        train_predictions,
        y_train_tensor
    )


    optimizer.zero_grad()


    train_loss.backward()


    optimizer.step()


    # ======================================
    # Validation
    # ======================================

    model.eval()


    with torch.no_grad():

        val_predictions = model(
            X_val_tensor
        )


        val_loss = loss_function(
            val_predictions,
            y_val_tensor
        )


    # ======================================
    # Store Loss
    # ======================================

    train_losses.append(
        train_loss.item()
    )


    val_losses.append(
        val_loss.item()
    )


    # ======================================
    # Save Best Model
    # ======================================

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        patience_counter = 0


        torch.save(
            model.state_dict(),
            "models/deep_learning/best_salary_nn.pth"
        )


    else:

        patience_counter += 1


    # ======================================
    # Print Progress
    # ======================================

    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f}"
        )


    # ======================================
    # Early Stopping
    # ======================================

    if patience_counter >= patience:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )

        break


# ==========================================
# 16. Load Best Model
# ==========================================

model.load_state_dict(
    torch.load(
        "models/deep_learning/best_salary_nn.pth",
        weights_only=True
    )
)


# ==========================================
# 17. Final Evaluation
# ==========================================

model.eval()


with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).numpy()


# ==========================================
# 18. Convert Predictions Back
# ==========================================

predictions = y_scaler.inverse_transform(
    predictions_scaled
)


# ==========================================
# 19. Evaluation Metrics
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)


r2 = r2_score(
    y_test,
    predictions
)


print("\n================================")
print("Deep Learning Salary Prediction")
print("================================")


print(
    f"MAE: ${mae:.2f}"
)


print(
    f"R2 Score: {r2:.4f}"
)


# ==========================================
# 20. Actual vs Predicted
# ==========================================

print("\nActual vs Predicted:")


for actual, predicted in zip(
    y_test.flatten(),
    predictions.flatten()
):

    print(
        f"Actual: ${actual:.0f} "
        f"| Predicted: ${predicted:.0f}"
    )


# ==========================================
# 21. Save Final Model
# ==========================================

torch.save(
    model.state_dict(),
    "models/deep_learning/salary_nn.pth"
)


# ==========================================
# 22. Plot Training vs Validation Loss
# ==========================================

plt.figure(figsize=(8, 5))


plt.plot(
    train_losses,
    label="Training Loss"
)


plt.plot(
    val_losses,
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)


plt.legend()

plt.grid(True)

plt.show()


# ==========================================
# 23. Final Output
# ==========================================

print(
    "\n================================"
)

print(
    "Files Saved Successfully"
)

print(
    "================================"
)


print(
    "Best Model:"
)

print(
    "models/deep_learning/best_salary_nn.pth"
)


print(
    "\nFinal Model:"
)

print(
    "models/deep_learning/salary_nn.pth"
)


print(
    "\nFeature Scaler:"
)

print(
    "models/deep_learning/X_scaler.pkl"
)


print(
    "\nSalary Scaler:"
)

print(
    "models/deep_learning/y_scaler.pkl"
)