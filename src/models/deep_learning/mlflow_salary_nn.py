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

import mlflow
import mlflow.pytorch

# ==========================================
# Reproducibility
# ==========================================

np.random.seed(42)
torch.manual_seed(42)

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
# 4. Scaling
# ==========================================

X_scaler = StandardScaler()

y_scaler = StandardScaler()


X_train = X_scaler.fit_transform(
    X_train
)

X_test = X_scaler.transform(
    X_test
)


y_train = y_scaler.fit_transform(
    y_train
)


# ==========================================
# 5. Validation Split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. Convert to PyTorch Tensors
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
# 7. Neural Network
# ==========================================

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


# ==========================================
# 8. Create Model
# ==========================================

model = SalaryNeuralNetwork()


# ==========================================
# 9. Loss + Optimizer
# ==========================================

loss_function = nn.MSELoss()


learning_rate = 0.01


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ==========================================
# 10. Training Configuration
# ==========================================

epochs = 500

patience = 30

best_val_loss = float("inf")

patience_counter = 0


train_losses = []

val_losses = []


# ==========================================
# 11. MLflow Experiment
# ==========================================

mlflow.set_experiment(
    "Career Salary Prediction"
)


# ==========================================
# 12. Start MLflow Run
# ==========================================

with mlflow.start_run(
    run_name="PyTorch Neural Network"
):


    # ======================================
    # Log Parameters
    # ======================================

    mlflow.log_param(
        "model",
        "PyTorch Neural Network"
    )

    mlflow.log_param(
        "input_features",
        9
    )

    mlflow.log_param(
        "hidden_layer_1",
        32
    )

    mlflow.log_param(
        "hidden_layer_2",
        16
    )

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    mlflow.log_param(
        "epochs",
        epochs
    )

    mlflow.log_param(
        "patience",
        patience
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )


    # ======================================
    # Training
    # ======================================

    for epoch in range(epochs):


        # ----------------------------------
        # Training
        # ----------------------------------

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


        # ----------------------------------
        # Validation
        # ----------------------------------

        model.eval()


        with torch.no_grad():

            val_predictions = model(
                X_val_tensor
            )


            val_loss = loss_function(
                val_predictions,
                y_val_tensor
            )


        train_loss_value = train_loss.item()

        val_loss_value = val_loss.item()


        train_losses.append(
            train_loss_value
        )

        val_losses.append(
            val_loss_value
        )


        # ----------------------------------
        # Log Loss Every Epoch
        # ----------------------------------

        mlflow.log_metric(
            "train_loss",
            train_loss_value,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss_value,
            step=epoch
        )


        # ----------------------------------
        # Best Model
        # ----------------------------------

        if val_loss_value < best_val_loss:

            best_val_loss = val_loss_value

            patience_counter = 0

            torch.save(
                model.state_dict(),
                "models/deep_learning/mlflow_best_salary_nn.pth"
            )

        else:

            patience_counter += 1


        # ----------------------------------
        # Progress
        # ----------------------------------

        if (epoch + 1) % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss_value:.4f} "
                f"Val Loss: {val_loss_value:.4f}"
            )


        # ----------------------------------
        # Early Stopping
        # ----------------------------------

        if patience_counter >= patience:

            print(
                f"\nEarly stopping at epoch {epoch + 1}"
            )

            break


    # ======================================
    # Load Best Model
    # ======================================

    model.load_state_dict(
        torch.load(
            "models/deep_learning/mlflow_best_salary_nn.pth",
            weights_only=True
        )
    )


    # ======================================
    # Test Prediction
    # ======================================

    model.eval()


    with torch.no_grad():

        predictions_scaled = model(
            X_test_tensor
        ).numpy()


    predictions = y_scaler.inverse_transform(
        predictions_scaled
    )


    # ======================================
    # Metrics
    # ======================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    print("\n==========================================")
    print("PyTorch Neural Network")
    print("==========================================")


    print(
        f"MAE: ${mae:.2f}"
    )


    print(
        f"R2 Score: {r2:.4f}"
    )


    # ======================================
    # Log Final Metrics
    # ======================================

    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "R2",
        r2
    )

    mlflow.log_metric(
        "best_validation_loss",
        best_val_loss
    )


    # ======================================
    # Create Loss Curve
    # ======================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        train_losses,
        label="Training Loss"
    )


    plt.plot(
        val_losses,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "PyTorch Training vs Validation Loss"
    )


    plt.legend()

    plt.grid(True)


    loss_plot_path = (
        "models/deep_learning/"
        "mlflow_loss_curve.png"
    )


    plt.savefig(
        loss_plot_path
    )


    plt.close()


    # ======================================
    # Log Loss Curve
    # ======================================

    mlflow.log_artifact(
        loss_plot_path
    )


    # ======================================
    # Log PyTorch Model
    # ======================================

    mlflow.pytorch.log_model(
    model,
    name="pytorch_salary_model",
    input_example=X_test_tensor[:1],
    serialization_format="pickle"
)


    # ======================================
    # Save Final Model
    # ======================================

    torch.save(
        model.state_dict(),
        "models/deep_learning/"
        "mlflow_salary_nn.pth"
    )


    # ======================================
    # Print Result
    # ======================================

    print(
        "\nExperiment logged to MLflow."
    )


# ==========================================
# 13. Finished
# ==========================================

print("\n==========================================")
print("PyTorch MLflow Tracking Complete")
print("==========================================")


print(
    "\nMLflow UI:"
)

print(
    "http://127.0.0.1:5000"
)