"""
Mission 3 -- Modelisation et comparaison rigoureuse.
Executer depuis la racine du projet : python src/train_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pipeline import load_raw, build_full_pipeline


def main():
    df = load_raw("data/Telco-Customer-Churn.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", class_weight="balanced", random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=== Validation croisee (memes plis pour tous les modeles) ===\n")
    for name, estimator in models.items():
        pipe = build_full_pipeline(estimator)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        results[name] = scores
        print(f"{name:20s} F1 = {scores.mean():.4f} +/- {scores.std():.4f}  |  plis: {np.round(scores,3)}")

    results_df = pd.DataFrame(results)
    print("\n=== Tableau recapitulatif ===")
    print(results_df.describe().T[["mean", "std"]])

    # Boxplot des scores par pli -- exige explicitement par le sujet (Mission 3)
    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    results_df.boxplot(ax=ax)
    ax.set_ylabel("F1-score par pli")
    ax.set_title("Comparaison des modeles -- validation croisee (5 plis)")
    plt.tight_layout()
    plt.savefig("reports/M3_boxplot_cv.png", dpi=120)
    plt.close()
    print("\nBoxplot sauvegarde : reports/M3_boxplot_cv.png")

    # Modele le plus stable = plus faible ecart-type
    most_stable = results_df.std().idxmin()
    print(f"\nModele le plus stable (plus faible ecart-type) : {most_stable}")

    # Test de Wilcoxon entre les 2 meilleurs modeles
    ranking = results_df.mean().sort_values(ascending=False)
    best_two = ranking.index[:2]
    print(f"\nDeux meilleurs modeles : {list(best_two)}")
    stat, pvalue = wilcoxon(results_df[best_two[0]], results_df[best_two[1]])
    print(f"Test de Wilcoxon {best_two[0]} vs {best_two[1]} : statistique={stat:.3f}, p-valeur={pvalue:.4f}")
    if pvalue < 0.05:
        print("=> Difference statistiquement significative (p < 0.05)")
    else:
        print("=> Difference NON significative (p >= 0.05) -- les modeles sont equivalents")

    # Entrainement du meilleur modele sur tout le train, analyse d'erreurs sur le test
    best_name = ranking.index[0]
    best_pipe = build_full_pipeline(models[best_name])
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    errors = X_test.copy()
    errors["y_true"] = y_test.values
    errors["y_pred"] = y_pred
    misclassified = errors[errors["y_true"] != errors["y_pred"]]

    print(f"\n=== Analyse d'erreurs ({best_name} sur le test) ===")
    print(f"Nombre de mal classes : {len(misclassified)} / {len(errors)}")
    print("\nRepartition Contract parmi les mal classes :")
    print(misclassified["Contract"].value_counts(normalize=True))
    print("\ntenure moyenne (mal classes vs tous) :")
    print(f"  mal classes : {misclassified['tenure'].mean():.1f}")
    print(f"  ensemble test : {errors['tenure'].mean():.1f}")


if __name__ == "__main__":
    main()