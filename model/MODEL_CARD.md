# Model Card — Prédiction de churn Telco

## Qu'est-ce qu'une model card et pourquoi c'est un standard
Une model card est une fiche technique standardisée qui documente un
modèle de machine learning : ses données d'entraînement, sa performance,
ses limites et ses cas d'usage prévus — introduite par Mitchell et al.
(2019, FAccT) pour que toute personne réutilisant un modèle (équipe
produit, auditeur, régulateur) comprenne son fonctionnement et ses
risques sans devoir relire tout le code. C'est devenu un standard car
il force une transparence minimale, en particulier sur l'équité entre
sous-groupes et les limites de généralisation — deux angles morts
fréquents des projets ML livrés sans documentation.

## Détails du modèle
- **Type** : régression logistique (scikit-learn), calibrée par
  `CalibratedClassifierCV` (méthode isotonique, 5 plis).
- **Version** : 1.0.0
- **Entraîné le** : voir date de génération de `model/pipeline_final.joblib`
- **Hyperparamètres principaux** : optimisés par Optuna (60 essais) —
  voir `reports/M4_optimisation.md` pour le détail complet.

## Données d'entraînement
- Telco Customer Churn (IBM Sample), ~7043 clients, split stratifié
  80/20 (5634 train / 1409 test).
- Période et origine géographique : dataset public de démonstration IBM,
  non daté précisément, origine américaine (télécom US) — **à noter
  comme limite** : les comportements de churn observés peuvent ne pas se
  généraliser directement à un autre marché ou une autre époque.
- Cible : `Churn` (résiliation, binaire), taux de base 26,5 %.

## Performance de validation
- F1 (CV 5 plis, train) : 0.636 ± variance selon les plis (voir M3/M4).
- Au seuil de décision retenu (0.119) sur le jeu de test : rappel ≈ 0.93,
  précision ≈ 0.44, F2 ≈ 0.75.
- Comparé à la baseline naïve (toujours "non-churn") : rappel 0 sur la
  classe churn — largement battue.

## Performance par sous-groupe (équité)
- **Non évaluée formellement dans cette V1** faute de variable protégée
  fiable dans le dataset (le genre est présent mais son usage comme
  feature de fairness nécessiterait une analyse dédiée, non réalisée
  ici par manque de temps sur ce capstone).
- **Limite reconnue** : avant tout déploiement réel, une analyse de
  demographic parity et d'equalized odds par sous-groupe (genre,
  ancienneté, type de contrat) est recommandée — voir question de
  réflexion 4 dans `reports/reflexion.md`.

## Limites connues
- Le modèle est entraîné sur un instantané figé ; il suppose la
  stationnarité des comportements de churn (voir Mission 0) — une
  évolution tarifaire ou concurrentielle peut le rendre obsolète sans
  réentraînement.
- Le seuil de décision (0.119) a été choisi pour un ratio de coûts
  FN/FP estimé (~15-20x) qui reste une approximation métier de la
  Mission 0, pas une valeur mesurée précisément — à recalibrer avec
  de vrais coûts opérationnels avant mise en production.
- Le dataset ne contient pas de signal temporel réel (pas d'historique
  multi-mois par client) — le modèle prédit un risque statique, pas une
  trajectoire.

## Usage prévu
Aide à la priorisation d'une campagne marketing de rétention. **Ne doit
pas** être utilisé comme seule justification pour refuser un service à
un client, ni comme preuve individuelle de désengagement.