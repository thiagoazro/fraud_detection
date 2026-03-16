import os
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/fraud-xgb@champion")

FEATURE_NAMES = ["amount", "hour", "dow", "channel", "international", "new_merchant",
                 "acct_age_days", "txn_count_24h", "txn_amount_24h", "distance_km",
                 "device_change", "ip_risk"]

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    yield

app = FastAPI(title="Fraud Detection API (XGBoost + MLflow)", lifespan=lifespan)

class PredictRequest(BaseModel):
    x: list[Union[list[float], dict]]  # batch: [[...]] ou [{feature: value}]

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI, "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if isinstance(req.x[0], dict):
        X = pd.DataFrame(req.x)
    else:
        X = pd.DataFrame(req.x, columns=FEATURE_NAMES)
    preds = model.predict(X)
    return {"predictions": preds.tolist()}
