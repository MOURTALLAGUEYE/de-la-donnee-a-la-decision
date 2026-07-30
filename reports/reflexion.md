# Questions de réflexion

## 1. Concept drift
Un modèle bon à l'entraînement peut se dégrader en production si la
relation entre les features et la cible change dans le temps, même si
le modèle lui-même ne change pas. Exemple ici : si un concurrent lance
une offre attractive spécifiquement pour les contrats `Two year`, ces
clients jusque-là "sûrs" pourraient se mettre à churner — le modèle
continuerait à leur attribuer un score de risque faible, à tort. Ce
qu'il faut surveiller : la performance réelle a posteriori (rappel,
précision sur les vraies résiliations du mois), pas seulement la
stabilité des données d'entrée — voir `reports/M5_monitoring.md`.

## 2. Information mutuelle vs corrélation de Pearson
La corrélation de Pearson ne mesure que les relations **linéaires**
entre deux variables numériques. L'information mutuelle est plus
générale : elle capture toute forme de dépendance statistique
(linéaire, non-linéaire, non-monotone) entre deux variables, y compris
catégorielles, sans supposer une forme particulière de relation. C'est
pourquoi elle a été utilisée en Mission 1 pour classer les features
(y compris catégorielles comme `Contract`) par pouvoir discriminant,
alors que Pearson n'aurait été applicable qu'aux 3 features numériques.

## 3. No Free Lunch
Le théorème No Free Lunch énonce qu'aucun algorithme n'est
universellement meilleur que tous les autres sur l'ensemble de tous
les problèmes possibles : la performance d'un modèle dépend
intrinsèquement de la structure des données auxquelles il est appliqué.
Cela justifie directement le benchmark comparatif de la Mission 3 :
on ne peut pas savoir a priori si un modèle linéaire (régression
logistique) ou un modèle à base d'arbres (forêt aléatoire) sera le
meilleur sur le churn Telco sans les tester tous les deux — et le
résultat obtenu (la régression logistique bat la forêt aléatoire par
défaut) illustre concrètement le théorème : la complexité
supplémentaire d'un modèle n'apporte pas automatiquement un gain.

## 4. Fairness
Le modèle n'a pas été formellement audité pour l'équité dans cette V1
(voir `model/MODEL_CARD.md`, limites connues). Deux métriques à
appliquer avant un déploiement réel :
- **Demographic parity** : le taux de clients prédits "à risque" doit
  être comparable entre sous-groupes (ex. par genre) — se mesure en
  comparant P(prédiction=churn | groupe A) à P(prédiction=churn |
  groupe B).
- **Equalized odds** : le taux de vrais positifs (rappel) et de faux
  positifs doit être comparable entre sous-groupes — se mesure en
  calculant séparément la matrice de confusion par sous-groupe et en
  comparant rappel et taux de faux positifs.
Ces deux métriques se calculent simplement avec le pipeline déjà en
place en filtrant `X_test`/`y_test` par sous-groupe avant d'appliquer
les métriques de la Mission 3.

## 5. Fuite de données — endroits vérifiés
Points précis du pipeline où une fuite aurait pu s'introduire, et
comment elle a été évitée :
- **Imputation/scaling/encodage** : auraient pu être calculés sur
  l'ensemble des données avant le split → évité en plaçant `train_test_split`
  strictement avant tout `fit` de transformation, et en encapsulant
  toutes les transformations dans un unique `Pipeline`/`ColumnTransformer`
  scikit-learn (`src/pipeline.py`), `fit` uniquement sur `X_train`.
- **Feature engineering** (`NumServices`, `ChargesPerTenure`) : aurait pu
  être calculé sur tout le dataset avant le split → évité en l'intégrant
  comme étape (`ServiceCountAdder`) du même `Pipeline`, appliquée après
  le split, de façon identique et sans ré-apprentissage sur train/test.
- **Validation croisée** : le tuning Optuna (Mission 4) aurait pu fuiter
  si le pipeline complet n'était pas ré-entraîné à chaque pli → évité en
  passant le `Pipeline` complet (pas seulement le modèle) à
  `cross_val_score`, garantissant que l'imputation/scaling sont
  recalculés à chaque pli sur les données d'entraînement du pli
  uniquement.
- **`customerID`** : variable sans pouvoir prédictif mais potentiellement
  corrélée à l'ordre de collecte → exclue dès `load_raw()` dans
  `src/pipeline.py`, avant toute étape du pipeline.
- **Calibration** : `CalibratedClassifierCV` aurait pu fuiter si calibré
  sur les mêmes données que l'entraînement du modèle → évité en utilisant
  son mode `cv=5` intégré, qui sépare automatiquement les folds
  d'entraînement du modèle de ceux utilisés pour ajuster la calibration.