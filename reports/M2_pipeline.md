# Mission 2 — Pipeline sans fuite de données

## Split
Split stratifié 80/20 : train = 5634 lignes (26,5 % churn), test = 1409
lignes (26,5 % churn) — proportions identiques, split correct.

## Pipeline
Un `ColumnTransformer` unique : sous-pipeline numérique (imputation
médiane + StandardScaler) et catégoriel (imputation mode + OneHotEncoder),
assemblé avec le feature engineering dans un seul `Pipeline` scikit-learn,
`fit` exclusivement sur le train (voir `src/pipeline.py`).

## Baseline
| Modèle | F1 (CV 5 plis) |
|---|---|
| Baseline naïve (classe majoritaire) | rappel = 0.000 sur la classe churn |
| Régression logistique, sans feature engineering | 0.633 ± 0.026 |
| Régression logistique, avec feature engineering | 0.629 ± 0.020 |

## Feature engineering
Deux features créées : `NumServices` (nombre de services "verrou"
souscrits) et `ChargesPerTenure` (ratio charges/ancienneté). Gain mesuré :
**-0.004 en F1**, non significatif au vu de l'écart-type. Conformément au
rasoir d'Occam demandé par le sujet, ces features linéaires n'apportent
rien à une régression logistique — elles sont conservées dans le pipeline
pour être réévaluées avec des modèles non-linéaires (Mission 3), mais
n'améliorent pas la baseline retenue.

## Accuracy finale sur le test (jamais touché avant cette évaluation)
0.743