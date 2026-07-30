# Mission 5 — Plan de monitoring en production

## Signaux à surveiller

1. **Data drift** (dérive des données d'entrée) : la distribution des
   features en production s'écarte-t-elle de celle du train ? Ex. si la
   proportion de clients en `Fiber optic` augmente fortement suite à une
   campagne commerciale, le modèle voit des profils sous-représentés
   à l'entraînement.
2. **Concept drift** (dérive de la relation features → cible) : le lien
   entre `Contract=Month-to-month` et le churn peut s'affaiblir si un
   concurrent lance une offre agressive qui change le comportement des
   clients engagés — la relation apprise devient obsolète même si les
   données d'entrée ne changent pas.
3. **Performance drift** : le F1/rappel réel constaté a posteriori (une
   fois qu'on sait qui a vraiment churné) se dégrade-t-il par rapport à
   la performance de validation ?

## Outil recommandé
**Evidently AI** (open-source) : génère des rapports de data drift
(test de Kolmogorov-Smirnov / PSI par feature) et de performance drift
en comparant un batch de référence (le train) à un batch de production,
avec alerting configurable.

## Fréquence proposée
- Data drift : hebdomadaire (batch marketing mensuel, mais surveillance
  plus fréquente pour détecter tôt).
- Performance drift : mensuel, dès que le vrai statut de churn des
  clients ciblés le mois précédent est connu.
- Réentraînement déclenché si : F1 réel < 0.55 sur 2 mois consécutifs,
  ou drift significatif détecté sur les 3 features dominantes
  (`tenure`, `Contract`, `ChargesPerTenure`).