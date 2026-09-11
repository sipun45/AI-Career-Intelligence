import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
import numpy as np
import joblib


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="Deep Learning Salary Prediction API"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Neural Network
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
# Load Model
# ==========================================

model = SalaryNeuralNetwork()

model.load_state_dict(
    torch.load(
        "models/deep_learning/best_salary_nn.pth",
        weights_only=True
    )
)

model.eval()


# ==========================================
# Load Scalers
# ==========================================

X_scaler = joblib.load(
    "models/deep_learning/X_scaler.pkl"
)

y_scaler = joblib.load(
    "models/deep_learning/y_scaler.pkl"
)


# ==========================================
# Input Schema
# ==========================================

class SalaryInput(BaseModel):

    experience: int

    education_encoded: int

    python: int

    java: int

    sql: int

    ml: int

    deep_learning: int

    cloud: int

    total_skills: int


# ==========================================
# Root Endpoint
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Deep Learning Salary Prediction API is running"
    }


# ==========================================
# Salary Prediction
# ==========================================

@app.post("/predict-salary-dl")
def predict_salary(data: SalaryInput):

    # --------------------------------------
    # Create feature array
    # --------------------------------------

    features = np.array([[
        data.experience,
        data.education_encoded,
        data.python,
        data.java,
        data.sql,
        data.ml,
        data.deep_learning,
        data.cloud,
        data.total_skills
    ]], dtype=np.float32)


    # --------------------------------------
    # Scale Features
    # --------------------------------------

    features_scaled = X_scaler.transform(
        features
    )


    # --------------------------------------
    # Convert to Tensor
    # --------------------------------------

    tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    with torch.no_grad():

        prediction_scaled = model(
            tensor
        ).numpy()


    # --------------------------------------
    # Convert Back to Salary
    # --------------------------------------

    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )


    predicted_salary = prediction[0][0]


    # --------------------------------------
    # Response
    # --------------------------------------

    return {
        "predicted_salary": round(
            float(predicted_salary),
            2
        )
    }