# %% [markdown]
# # Mission 1 — Données et analyse exploratoire
# Dataset : Telco Customer Churn (IBM Sample)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

df = pd.read_csv("../data/Telco-Customer-Churn.csv")
print(df.shape)
df.head()

# %% [markdown]
# ## 1. Profiling

# %%
print("Lignes / colonnes :", df.shape)
print("\nTypes de colonnes :\n", df.dtypes)

# %%
# TotalCharges est en texte : on regarde pourquoi avant de convertir
print("Valeurs non numériques dans TotalCharges :")
print(df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()][
    ["customerID", "tenure", "TotalCharges", "MonthlyCharges", "Churn"]
])

# %%
# Conversion propre : les valeurs vides deviennent NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Valeurs manquantes par colonne :\n", df.isna().sum()[df.isna().sum() > 0])
print(f"\n% manquant sur TotalCharges : {df['TotalCharges'].isna().mean()*100:.2f}%")

# %%
# Doublons
print("Lignes dupliquées :", df.duplicated().sum())
print("customerID dupliqués :", df["customerID"].duplicated().sum())

# %%
# Features quasi-constantes (une seule modalité >95% des cas)
for col in df.select_dtypes(include="object").columns:
    top_freq = df[col].value_counts(normalize=True).iloc[0]
    if top_freq > 0.95:
        print(f"{col} : quasi-constante ({top_freq:.1%} sur la modalité majoritaire)")

# %% [markdown]
# **Observation attendue** : les 11 valeurs manquantes de `TotalCharges`
# correspondent à des clients avec `tenure = 0` (nouveaux clients jamais
# facturés) — ce n'est pas une fuite, c'est une valeur manquante logique,
# à traiter par imputation dans le pipeline (Mission 2), jamais en la
# remplaçant "à la main" avant le split.

# %% [markdown]
# ## 2. Détection de fuite de données

# %%
# customerID n'a aucune valeur prédictive et doit être exclu du modèle
df_check = df.drop(columns=["customerID"]).copy()
df_check["Churn_bin"] = (df_check["Churn"] == "Yes").astype(int)

# Une feature qui "prédit trop bien" la cible est suspecte de fuite.
# On vérifie la corrélation des numériques avec la cible.
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
print(df_check[num_cols + ["Churn_bin"]].corr()["Churn_bin"])

# %% [markdown]
# **Vérification** : aucune feature du dataset n'est connue uniquement
# *après* le départ du client (pas de "date de résiliation", pas de
# "raison du départ" dans les colonnes). `customerID` est un identifiant
# pur, sans pouvoir prédictif — exclu du modèle. Aucune fuite détectée
# à ce stade.

# %% [markdown]
# ## 3. Analyse bivariée

# %%
# Taux de churn par variable catégorielle
cat_cols = df_check.select_dtypes(include="object").columns.drop("Churn")
for col in cat_cols:
    rate = df_check.groupby(col)["Churn_bin"].mean().sort_values(ascending=False)
    print(f"\n--- {col} ---\n{rate}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flat, ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]):
    sns.barplot(data=df_check, x=col, y="Churn_bin", ax=ax, errorbar=None)
    ax.set_ylabel("Taux de churn")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("../reports/M1_bivariate_categorical.png", dpi=120)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(data=df_check, x="Churn", y=col, ax=ax)
plt.tight_layout()
plt.savefig("../reports/M1_bivariate_numeric.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Pouvoir discriminant (information mutuelle)

# %%
df_mi = df_check.drop(columns=["Churn"]).copy()
le_dict = {}
for col in df_mi.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_mi[col] = le.fit_transform(df_mi[col].astype(str))
    le_dict[col] = le

df_mi["TotalCharges"] = df_mi["TotalCharges"].fillna(df_mi["TotalCharges"].median())

mi = mutual_info_classif(
    df_mi.drop(columns=["Churn_bin"]), df_mi["Churn_bin"], random_state=42
)
mi_series = pd.Series(mi, index=df_mi.drop(columns=["Churn_bin"]).columns).sort_values(ascending=False)
print("Top 5 features les plus prédictives (information mutuelle) :")
print(mi_series.head(5))

fig, ax = plt.subplots(figsize=(8, 6))
mi_series.head(10).plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("Information mutuelle")
plt.tight_layout()
plt.savefig("../reports/M1_mutual_information.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Hypothèses (à vérifier graphiquement ci-dessus)
#
# 1. Les clients en contrat mensuel (`Month-to-month`) churnent davantage
#    que ceux en contrat annuel/biannuel.
# 2. Les clients sans service de sécurité en ligne (`OnlineSecurity = No`)
#    churnent davantage — absence de "verrou" contractuel/technique.
# 3. Les nouveaux clients (`tenure` faible) churnent davantage que les
#    clients anciens — effet de fidélisation dans le temps.
# 4. Les clients payant par chèque électronique (`Electronic check`)
#    churnent davantage que les autres modes de paiement (moins engageant,
#    pas de prélèvement automatique).
# 5. Les clients avec `MonthlyCharges` élevé et peu de services groupés
#    (fibre seule, sans add-ons) churnent davantage — perception de
#    "cherté" sans valeur ajoutée perçue.
#
# Ces hypothèses guideront le feature engineering de la Mission 2
# (ex. compter le nombre de services souscrits, catégoriser l'ancienneté).

# %% [markdown]
# ## 6. Synthèse — 3 insights majeurs
#
# 1. **Le type de contrat est le facteur dominant** (MI la plus forte,
#    0.092) : les clients en `Month-to-month` churnent à 42,7 %, contre
#    11,3 % en engagement annuel et seulement 2,8 % en biannuel —
#    l'absence d'engagement contractuel est le principal levier de risque.
# 2. **L'ancienneté (`tenure`) est le deuxième facteur le plus prédictif**
#    (MI = 0.077, corrélation -0.35 avec le churn) : les nouveaux clients
#    partent nettement plus, ce qui suggère un enjeu d'onboarding/rétention
#    dans les premiers mois.
# 3. **Les services de "verrou" (`OnlineSecurity`, `TechSupport`,
#    `OnlineBackup`) réduisent fortement le churn** (~15% avec le service
#    contre ~42% sans) : un client équipé de plusieurs services est plus
#    ancré et moins volatile. Le mode de paiement `Electronic check` est
#    lui aussi un signal fort (45,3% de churn) — probablement lié à
#    l'absence de prélèvement automatique, donc moins d'engagement passif.