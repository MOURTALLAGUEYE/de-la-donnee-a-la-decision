# Mission 3 — Modélisation et comparaison

## Benchmark (StratifiedKFold, 5 plis, mêmes plis pour tous)
| Modèle | F1 moyen | Écart-type |
|---|---|---|
| Régression logistique | 0.632 | 0.024 |
| Forêt aléatoire (défaut) | 0.537 | 0.028 |
| SVM (RBF) | 0.630 | 0.019 |

Modèle le plus stable : **SVM (RBF)** (écart-type le plus faible).

## Test statistique
Wilcoxon entre les 2 meilleurs modèles (régression logistique vs SVM) :
p-valeur = 0.625 → **différence non significative**. Les deux modèles sont
statistiquement équivalents sur ce jeu de données à ce stade (avant tuning).

## Analyse d'erreurs
362 clients mal classés sur 1409 (25,7 %). Profil des mal classés :
83,4 % sont en contrat `Month-to-month` (contre ~55 % dans l'ensemble du
test), et leur ancienneté moyenne est plus faible (23,0 mois contre 31,9
pour l'ensemble). Le modèle confond donc principalement des clients en
contrat mensuel dont d'autres signaux (montant facturé, services
souscrits) suggèrent une fidélité que le contrat seul ne capture pas —
cela suggère d'explorer en Mission 4 une interaction entre `Contract` et
`NumServices`.

## Justification du choix des 3 familles
- **Régression logistique** : modèle linéaire, interprétable, bon
  point de référence pour un problème où plusieurs relations
  (contrat, ancienneté) semblent monotones.
- **Forêt aléatoire** : capture les interactions non-linéaires
  entre variables catégorielles nombreuses, robuste au surapprentissage
  par nature (bagging).
- **SVM (RBF)** : explore une frontière de décision non-linéaire
  différente de la forêt aléatoire, bon complément pour la comparaison.