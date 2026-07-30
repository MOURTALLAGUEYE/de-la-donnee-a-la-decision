# De la Donnée à la Décision
Projet final — Cours pratique de Supervised Learning, Master IA

## Problème
Prédire le churn (résiliation) des clients d'un opérateur télécom à partir
du dataset Telco Customer Churn (IBM Sample, ~7043 clients), afin de
cibler une campagne de rétention. Voir `reports/M0_cadrage.md` pour le
cadrage complet (coûts d'erreur, métrique, seuil de réussite).

## Données
- Source : Telco Customer Churn (IBM Sample), téléchargé automatiquement
  via `src/download_data.py`.
- ~7043 clients, 20 features (démographiques, contractuelles, services
  souscrits) + la cible `Churn`.
- Détails du profiling et de l'EDA : `reports/M1_bivariate_*.png`,
  `notebooks/01_eda.py`.

### Note sur la source des données
Le dataset est téléchargé automatiquement depuis un miroir GitHub
(IBM/telco-customer-churn-on-icp4d) strictement identique au dataset
Kaggle `blastchar/telco-customer-churn` référencé dans le sujet
(mêmes 7043 lignes, mêmes 21 colonnes) — utilisé pour permettre un
téléchargement scripté reproductible (`src/download_data.py`) sans
authentification Kaggle.

## Modèle
- Pipeline scikit-learn unique (`src/pipeline.py`) : feature engineering
  (`NumServices`, `ChargesPerTenure`) + imputation + scaling + one-hot
  encoding, `fit` exclusivement sur le train — aucune fuite de données.
- Régression logistique, optimisée par Optuna (60 essais, avec
  MedianPruner, `src/tune_and_explain.py`), calibrée
  (`CalibratedClassifierCV`, méthode isotonique).
- Performance : F1 (CV 5 plis) ≈ 0.636 ; seuil de décision optimisé
  (F2) ≈ 0.119, rappel ≈ 0.93 sur la classe churn.
- Détails complets : `reports/M2_pipeline.md`, `reports/M3_benchmark.md`,
  `reports/M4_optimisation.md`.

## Structure du dépôt
## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --only-binary=:all: -r requirements.txt
```

## Reproduire le pipeline complet

```powershell
python src\download_data.py       # Mission 1 — télécharge le dataset
python src\train_baseline.py      # Mission 2 — pipeline + baseline
python src\train_models.py        # Mission 3 — benchmark 3 modèles
python src\tune_and_explain.py    # Mission 4 — tuning, calibration, SHAP
                                   # -> sauvegarde model/pipeline_final.joblib
pytest tests\ -v                  # Mission 5 — tests
```

## Lancer l'API

```powershell
uvicorn api.main:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

Endpoints :
- `GET /health` — statut et version du service
- `POST /predict` — prédiction de churn pour un client (JSON)
- `GET /model-info` — métadonnées et performance du modèle

## Monitoring en production
Voir `reports/M5_monitoring.md` pour les signaux à surveiller
(data drift, concept drift, performance drift) et les outils recommandés.

## Reproductibilité
`random_state=42` fixé partout (split, modèles, validation croisée,
Optuna). Le pipeline sérialisé est rechargé et testé automatiquement
(`tests/test_pipeline.py::test_reloaded_pipeline_is_deterministic`).

## Auteur
MOURTALLA GUEYE — Master 1 DSIA, ISI — Année universitaire 2025–2026