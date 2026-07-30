"""
Pipeline de préparation des données — Mission 2.

Point noté le plus strictement : AUCUNE transformation apprise (imputation,
encodage, scaling, feature engineering statistique) ne doit être calculée
sur autre chose que le train. Tout vit dans un objet Pipeline scikit-learn.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_raw(path: str) -> pd.DataFrame:
    """Charge le CSV brut et fait UNIQUEMENT les corrections non apprises
    (typage, pas de statistique calculée sur les données)."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)  # 0/1 -> catégorielle
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df


class ServiceCountAdder(BaseEstimator, TransformerMixin):
    """Feature engineering : nombre de services 'verrou' souscrits.
    Compatible Pipeline scikit-learn (fit/transform), n'apprend aucune
    statistique du jeu de données -- juste un comptage ligne à ligne."""

    lock_in_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["NumServices"] = (X[self.lock_in_cols] == "Yes").sum(axis=1)
        X["ChargesPerTenure"] = X["MonthlyCharges"] / X["tenure"].replace(0, 1)
        return X


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES + ["NumServices", "ChargesPerTenure"]),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def build_full_pipeline(estimator) -> Pipeline:
    """Assemble feature engineering + preprocessing + modèle en UN SEUL
    Pipeline, fit exclusivement sur le train."""
    return Pipeline(steps=[
        ("feature_engineering", ServiceCountAdder()),
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])