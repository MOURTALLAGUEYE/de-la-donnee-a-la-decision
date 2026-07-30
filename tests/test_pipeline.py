"""
Mission 5 -- Tests pytest.
Executer depuis la racine du projet : pytest tests/ -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline import load_raw

MODEL_PATH = ROOT / "model" / "pipeline_final.joblib"
THRESHOLD_PATH = ROOT / "model" / "decision_threshold.joblib"
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def threshold():
    return joblib.load(THRESHOLD_PATH)


@pytest.fixture(scope="module")
def sample_data():
    df = load_raw(str(DATA_PATH))
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X.head(50), y.head(50)


def test_output_shape(pipeline, sample_data):
    """La sortie a la bonne forme : une probabilite par ligne d'entree."""
    X, _ = sample_data
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_probabilities_in_range(pipeline, sample_data):
    """Les probabilites predites sont bien dans [0, 1]."""
    X, _ = sample_data
    proba = pipeline.predict_proba(X)[:, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_handles_missing_values(pipeline, sample_data):
    """Le pipeline gere les valeurs manquantes sans planter (imputation)."""
    X, _ = sample_data
    X_missing = X.copy()
    X_missing.loc[X_missing.index[0], "TotalCharges"] = np.nan
    X_missing.loc[X_missing.index[1], "OnlineSecurity"] = np.nan
    proba = pipeline.predict_proba(X_missing)
    assert not np.isnan(proba).any()


def test_expected_features_present(sample_data):
    """Les colonnes attendues par le pipeline sont bien presentes."""
    X, _ = sample_data
    expected = {
        "tenure", "MonthlyCharges", "TotalCharges", "Contract",
        "InternetService", "PaymentMethod", "gender",
    }
    assert expected.issubset(set(X.columns))


def test_performance_on_reference_set(pipeline, threshold, sample_data):
    """La performance sur un mini jeu de reference reste correcte
    (le rappel sur la classe churn doit rester raisonnable)."""
    X, y = sample_data
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    y_bin = y.values
    if y_bin.sum() > 0:
        recall = ((preds == 1) & (y_bin == 1)).sum() / y_bin.sum()
        assert recall >= 0.3  # seuil bas car mini-echantillon de 50 lignes


def test_reloaded_pipeline_is_deterministic(sample_data):
    """Un pipeline recharge produit des predictions strictement identiques
    a un autre rechargement (reproductibilite de la serialisation)."""
    X, _ = sample_data
    pipe_a = joblib.load(MODEL_PATH)
    pipe_b = joblib.load(MODEL_PATH)
    proba_a = pipe_a.predict_proba(X)[:, 1]
    proba_b = pipe_b.predict_proba(X)[:, 1]
    assert np.allclose(proba_a, proba_b, atol=1e-12)