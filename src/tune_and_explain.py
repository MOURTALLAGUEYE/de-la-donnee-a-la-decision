"""
Mission 4 -- Optimisation (Optuna), calibration, interpretabilite (SHAP),
choix du seuil de decision.
Executer depuis la racine du projet : python src/tune_and_explain.py
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import fbeta_score, precision_recall_curve

from pipeline import load_raw, build_full_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    df = load_raw("data/Telco-Customer-Churn.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------- 1. Tuning Optuna (regression logistique, 6+ hyperparametres, avec pruner) ----------
    from sklearn.metrics import f1_score

    def objective(trial):
        params = dict(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="liblinear",  # supporte l1 et l2
            class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
            max_iter=trial.suggest_int("max_iter", 500, 3000, step=500),
            tol=trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
            intercept_scaling=trial.suggest_float("intercept_scaling", 0.1, 5.0),
        )
        model = LogisticRegression(random_state=42, **params)
        pipe = build_full_pipeline(model)

        # Boucle manuelle sur les plis pour pouvoir reporter un score
        # intermediaire a Optuna a chaque pli -> necessaire pour que le
        # pruner (MedianPruner) puisse reellement arreter un essai peu
        # prometteur avant d'avoir teste tous les plis.
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            fold_scores.append(f1_score(y_val, pred))

            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    print("=== Tuning Optuna (60 essais, avec MedianPruner) ===")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=60, show_progress_bar=False)

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Essais completes : {n_complete} | Essais elagues (pruned) : {n_pruned}")

    print(f"Meilleur F1 (CV) : {study.best_value:.4f}")
    print("Meilleurs hyperparametres :")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    default_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    default_pipe = build_full_pipeline(default_model)
    default_scores = cross_val_score(default_pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"\nF1 CV avec parametres par defaut : {default_scores.mean():.4f}")
    print(f"Gain du tuning : {study.best_value - default_scores.mean():+.4f}")

    importances = optuna.importance.get_param_importances(study)
    print("\nImportance des hyperparametres :")
    for k, v in importances.items():
        print(f"  {k} : {v:.3f}")

    # ---------- 2. Entrainer le modele final optimise ----------
    best_params = dict(study.best_params)
    best_params["solver"] = "liblinear"
    best_model = LogisticRegression(random_state=42, **best_params)
    best_pipe = build_full_pipeline(best_model)
    best_pipe.fit(X_train, y_train)

    # ---------- 3. Calibration ----------
    proba_uncalibrated = best_pipe.predict_proba(X_test)[:, 1]
    frac_pos_before, mean_pred_before = calibration_curve(y_test, proba_uncalibrated, n_bins=10)

    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    proba_calibrated = calibrated.predict_proba(X_test)[:, 1]
    frac_pos_after, mean_pred_after = calibration_curve(y_test, proba_calibrated, n_bins=10)

    print("\n=== Calibration ===")
    print("Avant calibration (fraction reelle vs predite par bin) :")
    for fp, mp in zip(frac_pos_before, mean_pred_before):
        print(f"  predit={mp:.2f}  reel={fp:.2f}")
    print("Apres calibration (isotonic) :")
    for fp, mp in zip(frac_pos_after, mean_pred_after):
        print(f"  predit={mp:.2f}  reel={fp:.2f}")

    err_before = np.mean(np.abs(frac_pos_before - mean_pred_before))
    err_after = np.mean(np.abs(frac_pos_after - mean_pred_after))
    print(f"\nErreur de calibration moyenne avant : {err_before:.4f}")
    print(f"Erreur de calibration moyenne apres  : {err_after:.4f}")

    # Reliability diagram (graphique) -- explicitement exige par le sujet
    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibration parfaite")
    ax.plot(mean_pred_before, frac_pos_before, marker="o", label="Avant calibration")
    ax.plot(mean_pred_after, frac_pos_after, marker="s", label="Apres calibration (isotonic)")
    ax.set_xlabel("Probabilite predite moyenne (par bin)")
    ax.set_ylabel("Fraction reelle de churn (par bin)")
    ax.set_title("Reliability diagram")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/M4_reliability_diagram.png", dpi=120)
    plt.close()
    print("Reliability diagram sauvegarde : reports/M4_reliability_diagram.png")

    # ---------- 4. Choix du seuil (cout metier Mission 0) ----------
    # Cout FN ~= 900, cout FP ~= 40 (valeurs M0) -> on maximise F-beta avec
    # beta ~ sqrt(900/40) ~ 4.7, on utilise F2 comme proxy raisonnable
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_calibrated)
    beta = 2.0
    f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-9)
    best_idx = np.argmax(f_beta_scores[:-1])
    best_threshold = thresholds[best_idx]

    print(f"\n=== Choix du seuil ===")
    print(f"Seuil optimal (F2) : {best_threshold:.3f}")
    print(f"F2 a ce seuil : {f_beta_scores[best_idx]:.4f}")
    print(f"Precision : {precisions[best_idx]:.3f} | Rappel : {recalls[best_idx]:.3f}")

    f2_at_05 = fbeta_score(y_test, (proba_calibrated >= 0.5).astype(int), beta=2)
    print(f"F2 au seuil par defaut (0.5) : {f2_at_05:.4f}")

    # ---------- 5. Interpretabilite SHAP ----------
    print("\n=== SHAP ===")
    # SHAP a besoin du pipeline SANS calibration (modele lineaire brut),
    # on explique le pipeline optimise best_pipe (pre-calibration).
    preprocessor = best_pipe.named_steps["preprocessor"]
    fe = best_pipe.named_steps["feature_engineering"]
    model = best_pipe.named_steps["model"]

    X_test_fe = fe.transform(X_test)
    X_test_transformed = preprocessor.transform(X_test_fe)
    feature_names = preprocessor.get_feature_names_out()

    X_test_dense = X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed
    X_test_df = pd.DataFrame(X_test_dense, columns=feature_names)

    explainer = shap.LinearExplainer(model, X_test_df)
    shap_values = explainer(X_test_df)

    # Importance globale (top 10)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    print("Top 10 features (importance SHAP globale) :")
    print(importance_df.head(10))

    # Summary plot
    Path("reports").mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig("reports/M4_shap_summary.png", dpi=120)
    plt.close()

    # 3 decisions individuelles : 1 vrai positif, 1 vrai negatif, 1 faux positif
    proba_test = model.predict_proba(preprocessor.transform(fe.transform(X_test)))[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    y_test_arr = y_test.values

    idx_tp = np.where((pred_test == 1) & (y_test_arr == 1))[0]
    idx_tn = np.where((pred_test == 0) & (y_test_arr == 0))[0]
    idx_fp = np.where((pred_test == 1) & (y_test_arr == 0))[0]

    for label, idx_array in [("vrai_positif", idx_tp), ("vrai_negatif", idx_tn), ("faux_positif", idx_fp)]:
        if len(idx_array) == 0:
            continue
        i = idx_array[0]
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        plt.tight_layout()
        plt.savefig(f"reports/M4_shap_{label}.png", dpi=120)
        plt.close()
        print(f"Explication {label} sauvegardee : reports/M4_shap_{label}.png")

    # Dependence plot de la feature dominante
    top_feature = importance_df.index[0]
    plt.figure()
    shap.dependence_plot(top_feature, shap_values.values, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig("reports/M4_shap_dependence.png", dpi=120)
    plt.close()
    print(f"Dependence plot sauvegarde pour : {top_feature}")

    # ---------- 6. Sauvegarde des objets pour Mission 5 ----------
    import joblib
    Path("model").mkdir(exist_ok=True)
    joblib.dump(calibrated, "model/pipeline_final.joblib")
    joblib.dump(best_threshold, "model/decision_threshold.joblib")
    print("\nModele calibre et seuil sauvegardes dans model/")


if __name__ == "__main__":
    main()