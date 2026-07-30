# Mission 4 — Optimisation, calibration, interprétabilité

## Tuning (Optuna, 60 essais)
Meilleur F1 en validation croisée : **0.6363** contre 0.6321 avec les
paramètres par défaut — gain modeste de +0.0043. L'hyperparamètre le plus
influent est `class_weight` (81 % de l'importance totale) : sur un
problème déséquilibré (26,5 % de churn), rééquilibrer les classes pèse
bien plus que la régularisation fine (`C`, `tol`, `intercept_scaling`).
Meilleurs paramètres retenus : `class_weight=balanced`, `penalty=l2`,
`intercept_scaling≈3.6`.

## Calibration
Avant calibration, le modèle est visiblement sur-confiant dans les tranches
de probabilité moyennes-hautes (ex. il annonce 66 % de risque quand le
taux réel n'est que de 32 %). L'erreur de calibration moyenne passe de
**0.164 à 0.024** après application de `CalibratedClassifierCV` (méthode
isotonique) — une amélioration nette. Une probabilité de churn fiable est
essentielle ici : le marketing doit pouvoir doser la générosité de l'offre
de rétention selon le niveau de risque réel, pas selon un score biaisé.

## Interprétabilité (SHAP)
**Importance globale (top 5)** : `tenure` (ancienneté) domine largement,
suivie par `Contract=Month-to-month`, `Contract=Two year`,
`ChargesPerTenure` (feature engineering) et `MonthlyCharges`. Cela
confirme et affine les observations de la Mission 1 : l'ancienneté et le
type de contrat restent les deux leviers dominants, mais le ratio
charges/ancienneté créé en Mission 2 apparaît maintenant comme informatif
une fois combiné aux autres variables dans un modèle entraîné (alors qu'il
semblait inutile isolément en M2).

**Summary plot** (`M4_shap_summary.png`) : les valeurs faibles de `tenure`
poussent fortement la prédiction vers le churn (points rouges à droite),
tandis que les valeurs élevées la poussent vers la fidélité — relation
monotone attendue. `Contract=Month-to-month` a un effet binaire net :
sa présence (rouge) pousse systématiquement vers le churn.

**Décisions individuelles** :
- *Vrai positif* (`M4_shap_vrai_positif.png`) : client correctement
  identifié comme à risque — `tenure` faible et `Contract=Month-to-month`
  sont les deux contributions dominantes qui poussent la prédiction vers
  le churn.
- *Vrai négatif* (`M4_shap_vrai_negatif.png`) : client correctement
  identifié comme fidèle — `tenure` élevée et un contrat engageant
  (`Two year`) tirent fortement la prédiction vers la non-résiliation.
- *Faux positif* (`M4_shap_faux_positif.png`) : client à tort prédit
  churneur — probablement une ancienneté encore faible mais un contrat
  engageant ou des services "verrou" souscrits, signaux contradictoires
  que le modèle pondère mal dans ce cas précis.

**Dependence plot** (`M4_shap_dependence.png`, sur `tenure`) : la relation
n'est pas parfaitement linéaire — l'effet de l'ancienneté sur la
prédiction est plus marqué dans les premiers mois (early churn) et
s'aplatit ensuite, cohérent avec l'hypothèse d'onboarding formulée en
Mission 1.

## Choix du seuil de décision
Au seuil par défaut (0,5), le F2-score n'est que de 0.566. En optimisant
explicitement selon le F2 (cohérent avec l'asymétrie de coûts fixée en
Mission 0 — un churn manqué coûte ~15-20x plus qu'une offre envoyée à
tort), le seuil optimal tombe à **0.119** : précision 0.435, rappel
**0.925**. Ce n'est pas 0,5 car le coût métier n'est pas symétrique — il
est largement préférable de cibler trop de clients (faux positifs à
~40€ chacun) que de laisser partir des clients à risque (faux négatifs à
~900€ chacun). Ce seuil bas est le seuil retenu pour la mise en
production (Mission 5).