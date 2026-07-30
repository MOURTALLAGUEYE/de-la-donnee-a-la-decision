"""
Mission 5 -- API REST (FastAPI).
Lancer depuis la racine du projet : uvicorn api.main:app --reload
Documentation interactive une fois lancee : http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = ROOT / "model" / "pipeline_final.joblib"
THRESHOLD_PATH = ROOT / "model" / "decision_threshold.joblib"
VERSION = "1.0.0"

app = FastAPI(
    title="API de prediction de churn -- De la Donnee a la Decision",
    version=VERSION,
)

pipeline = None
threshold = None


@app.on_event("startup")
def load_model():
    global pipeline, threshold
    pipeline = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)


class ClientFeatures(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: str = Field(examples=["0"])
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[12], ge=0)
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["Yes"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[75.5], ge=0)
    TotalCharges: float | None = Field(default=None, examples=[906.0])


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    decision_threshold: float


@app.get("/health")
def health():
    """Statut du service et version du modele."""
    return {"status": "ok", "version": VERSION, "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    """Predit la probabilite de churn pour un client donne."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    data = pd.DataFrame([client.model_dump()])
    try:
        proba = pipeline.predict_proba(data)[0, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Entree invalide : {exc}")
    return PredictionResponse(
        churn_prediction=bool(proba >= threshold),
        churn_probability=round(float(proba), 4),
        decision_threshold=round(float(threshold), 4),
    )


@app.get("/model-info")
def model_info():
    """Features attendues, metadonnees et performance de validation."""
    return {
        "version": VERSION,
        "model_type": "LogisticRegression (calibree, isotonic)",
        "decision_threshold": round(float(threshold), 4) if threshold else None,
        "expected_features": list(ClientFeatures.model_fields.keys()),
        "training_dataset": "Telco Customer Churn (IBM Sample, ~7043 clients)",
        "validation_metric": "F1 (CV 5 plis) ~= 0.636 ; F2 au seuil retenu ~= 0.75",
    }