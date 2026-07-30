# Mission 0 — Cadrage du problème

## Problème métier
L'opérateur télécom souhaite identifier, avant leur résiliation, les clients à
risque de churn afin que le **service marketing** puisse leur adresser en
priorité une **offre de rétention personnalisée**. La variable `Churn` est
définie comme : le client a résilié son contrat au cours de la période
d'observation (`Churn = Yes`) ou non (`Churn = No`).

## Coûts d'erreur (estimation, en euros)
- **Faux négatif** (client qui part, non détecté) : perte de la valeur vie
  client restante. Avec un ARPU moyen ~65 €/mois et une durée de rétention
  moyenne estimée à ~18 mois si retenu, on estime le coût d'un churn manqué
  à environ **800–1000 €** (valeur future perdue + coût d'acquisition d'un
  nouveau client pour compenser, généralement 5x plus cher que la rétention).
- **Faux positif** (offre envoyée à un client fidèle) : coût de la remise
  commerciale (ex. -20 % pendant 3 mois) ≈ **30–50 €**, plus un risque
  marginal de banaliser les offres.
- **Conclusion** : le faux négatif coûte environ **15 à 20 fois plus cher**
  que le faux positif. Ce déséquilibre justifie de privilégier le
  **rappel (recall)** de la classe churn, quitte à accepter davantage de
  faux positifs.

## Métrique
- **Métrique principale** : F2-score (pondère le rappel 2x plus que la
  précision) sur la classe positive (`Churn = Yes`), cohérent avec
  l'asymétrie de coûts ci-dessus.
- **Métriques secondaires** : ROC-AUC (robustesse globale du classement) et
  précision de la classe churn (pour ne pas saturer les équipes marketing
  de faux positifs).
- **Seuil de réussite fixé a priori** : F2-score ≥ 0.65 et rappel ≥ 0.75 sur
  le jeu de test, en battant significativement la baseline naïve (toujours
  prédire "non-churn", qui a un rappel de 0 sur la classe churn).

## Risques & hypothèses
- **Stationnarité** : on suppose que les comportements de churn observés
  restent globalement stables à court terme ; un changement tarifaire ou une
  offre concurrente peut casser cette hypothèse (→ monitoring, Mission 5).
- **Latence** : usage principal en batch mensuel (pas de contrainte temps
  réel forte) ; l'API doit néanmoins répondre en < 200 ms pour un usage
  ponctuel (CRM, simulateur).
- **RGPD** : données pseudonymisées (`customerID`), aucune donnée directement
  identifiante utilisée comme feature.
- **Explicabilité** : requise pour que le marketing comprenne *pourquoi* un
  client est ciblé (→ SHAP, Mission 4).