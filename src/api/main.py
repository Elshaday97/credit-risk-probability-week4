from fastapi import FastAPI
import pandas as pd
from .model_loader import load_model
from .pydantic_models import PredictionRequest, PredictionResponse
from scripts.constants import Aggregated_Columns, MODEL_NAME
import os
import mlflow
from pathlib import Path

project_root = Path.cwd().parent

"""Initialize FastAPI app"""
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict customer credit risk using MLFlow deployed model",
    version="1.0",
)

""" Load the MLFlow model """
model = load_model()


FEATURE_ORDER = [
    Aggregated_Columns.TransactionCount.value,
    Aggregated_Columns.TotalTransactionAmount.value,
    Aggregated_Columns.UniqueProductCategoryCount.value,
    Aggregated_Columns.TransactionAmountSTD.value,
    Aggregated_Columns.AverageTransactionHour.value,
    Aggregated_Columns.AverageTransactionAmount.value,
    Aggregated_Columns.MostCommonChannel.value,
    Aggregated_Columns.MostCommonTransactionDay.value,
    Aggregated_Columns.MostCommonTransactionMonth.value,
]


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    """
    Predict credit risk probability for a single customer
    """

    input_df = pd.DataFrame([request.dict()])

    input_df = input_df[FEATURE_ORDER]
    print(input_df)

    # Predict probability
    risk_probability = model.predict(input_df)[0]

    return PredictionResponse(
        risk_probability=float(risk_probability),
        is_high_risk=int(risk_probability >= 0.5),
    )
