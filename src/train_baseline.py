"""
Mission 2 — Baseline et gain du feature engineering.
Exécuter depuis la racine du projet : python src/train_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from pipeline import load_raw, build_full_pipeline
from sklearn.pipeline import Pipeline


def main():
    df = load_raw("data/Telco-Customer-Churn.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # SPLIT D'ABORD -- avant toute imputation/encodage/scaling.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f"Taux de churn train : {y_train.mean():.3f} | test : {y_test.mean():.3f}")

    # --- Baseline naïve (classe majoritaire) ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_recall = (dummy.predict(X_test) == y_test)[y_test == 1].mean()
    print(f"\nBaseline naïve -- rappel classe churn : {dummy_recall:.3f} (doit être ~0)")

    # --- Régression logistique, SANS feature engineering ---
    pipe_no_fe = Pipeline(steps=[
        ("preprocessor", build_preprocessor_without_new_features()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    scores_no_fe = cross_val_score(pipe_no_fe, X_train, y_train, cv=5, scoring="f1")
    print(f"\nSans feature engineering -- F1 CV : {scores_no_fe.mean():.3f} +/- {scores_no_fe.std():.3f}")

    # --- Régression logistique, AVEC feature engineering (pipeline complet) ---
    pipe_fe = build_full_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    scores_fe = cross_val_score(pipe_fe, X_train, y_train, cv=5, scoring="f1")
    print(f"Avec feature engineering -- F1 CV : {scores_fe.mean():.3f} +/- {scores_fe.std():.3f}")

    gain = scores_fe.mean() - scores_no_fe.mean()
    print(f"\nGain du feature engineering (F1) : {gain:+.4f}")

    # Entraînement final sur tout le train, évaluation sur le test (jamais touché)
    pipe_fe.fit(X_train, y_train)
    test_score = pipe_fe.score(X_test, y_test)
    print(f"\nAccuracy pipeline final sur le TEST (jamais vu) : {test_score:.3f}")


def build_preprocessor_without_new_features():
    """Même préprocesseur mais sans les 2 features créées, pour mesurer
    honnêtement le gain du feature engineering."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    numeric_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    main()