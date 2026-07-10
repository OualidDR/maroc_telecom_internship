"""
5G-NIDD - Exploration complete du dataset
==========================================
Ce script regroupe TOUTES les explorations faites depuis le debut du projet,
plus les verifications restantes avant de figer contracts/schema.py.

Usage:
    python3 explor_full.py
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PATH = "data/raw/Combined.csv"


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ----------------------------------------------------------------------------
# 0. CHARGEMENT
# ----------------------------------------------------------------------------
section("0. CHARGEMENT DU DATASET")

# low_memory=False evite le DtypeWarning rencontre au tout debut (colonne dDSb
# avec types mixtes a cause du melange NaN / string)
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Shape:", df.shape)  # attendu: (1215890, 52)
print("\nDtypes:")
print(df.dtypes)
print("\nApercu:")
print(df.head(10))


# ----------------------------------------------------------------------------
# 1. LABELS ET DESEQUILIBRE DE CLASSES
# ----------------------------------------------------------------------------
section("1. DISTRIBUTION DES LABELS")

print("\nLabel (binaire):")
print(df["Label"].value_counts())

print("\nAttack Type (multiclasse, 9 classes):")
print(df["Attack Type"].value_counts())

print("\nAttack Tool (colonne NON documentee dans le brief initial):")
print(df["Attack Tool"].value_counts())


# ----------------------------------------------------------------------------
# 2. VALEURS MANQUANTES (structurelles, pas aleatoires)
# ----------------------------------------------------------------------------
section("2. VALEURS MANQUANTES")

nulls = df.isnull().sum().sort_values(ascending=False)
print(nulls)

# Verification: le NaN des colonnes d* correle avec l'absence de reponse dst
# et le NaN des colonnes TCP correle avec Proto != tcp
print("\nTaux de NaN par protocole (SrcWin = champ TCP-only, dTtl = champ 'dst'):")
print(df.groupby("Proto")[["SrcWin", "dTtl"]].apply(lambda g: g.isnull().mean()))


# ----------------------------------------------------------------------------
# 3. COLONNES REDONDANTES (Mean/Sum/Min/Max == Dur)
# ----------------------------------------------------------------------------
section("3. COLONNES POTENTIELLEMENT REDONDANTES")

cols_dur = ["Dur", "RunTime", "Mean", "Sum", "Min", "Max"]
print("Nunique par colonne:")
print(df[cols_dur].nunique())
print("\nFraction de lignes ou Mean == Dur:", (df["Dur"] == df["Mean"]).mean())
print("Fraction de lignes ou Sum == Dur:", (df["Dur"] == df["Sum"]).mean())
print("Fraction de lignes ou Min == Dur:", (df["Dur"] == df["Min"]).mean())
print("Fraction de lignes ou Max == Dur:", (df["Dur"] == df["Max"]).mean())


# ----------------------------------------------------------------------------
# 4. RECHERCHE D'UN SIGNAL TEMPOREL / CHRONOLOGIQUE (Seq)
# ----------------------------------------------------------------------------
section("4. ANALYSE DE LA COLONNE Seq")

resets = df.index[df["Seq"].diff() < 0].tolist()
print("Nombre de resets de Seq:", len(resets))
print("10 premiers indices de reset:", resets[:10])

bounds = [0] + resets + [len(df)]
seg_lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
print("\nStats sur la taille des segments entre 2 resets:")
print(pd.Series(seg_lens).describe())

df["segment_id"] = pd.Series(range(len(df))).isin(resets).cumsum()
mixed_segments = df.groupby("segment_id")["Attack Type"].nunique()
print(
    "\nSegments contenant plus d'un Attack Type:",
    (mixed_segments > 1).sum(),
    "/",
    mixed_segments.shape[0],
)

# CONCLUSION deja etablie: Seq n'est PAS un identifiant chronologique global.
# Il se reinitialise ~128 000 fois (bookkeeping interne Argus), sans lien
# avec les sessions d'attaque.


# ----------------------------------------------------------------------------
# 5. ORDRE DES LIGNES == ORDRE CHRONOLOGIQUE ? (blocs Attack Type)
# ----------------------------------------------------------------------------
section("5. BLOCS CONTIGUS DE Attack Type DANS L'ORDRE DES LIGNES")

blocks = df["Attack Type"].ne(df["Attack Type"].shift()).cumsum()
print("Nombre de blocs contigus:", blocks.nunique())
print("\nNombre de blocs par type d'attaque:")
print(df.groupby(blocks)["Attack Type"].first().value_counts())

block_sizes = df.groupby(blocks).size()
print("\nStats sur la taille des blocs:")
print(block_sizes.describe())

# CONCLUSION deja etablie: 21 121 blocs, taille mediane = 2 lignes.
# Les types d'attaques sont eparpilles dans tout le fichier.
# => L'ordre des lignes NE reflete PAS l'ordre chronologique reel.
# => Le simulateur de replay devra melanger les lignes (pas de vrai "temps reel"
#    reconstituable), en conservant les proportions de classes.


# ----------------------------------------------------------------------------
# 6. CE QU'IL RESTE A FAIRE (pas encore execute)
# ----------------------------------------------------------------------------

section("6. DOUBLONS")
print("Nombre de lignes dupliquees:", df.duplicated().sum())


section("7. VALEURS SENTINELLES CACHEES (min/max de chaque colonne numerique)")
for col in df.select_dtypes("number").columns:
    print(f"{col:15s} min={df[col].min():>15}  max={df[col].max():>15}")
# A surveiller particulierement: SrcTCPBase / DstTCPBase qui montent tres haut
# (~4.29 milliards) -> verifier si ce sont de vrais TCP sequence numbers
# ou une valeur sentinelle (ex: max uint32 = 4294967295 = "non applicable").


section("8. CARDINALITE DES COLONNES CATEGORIELLES")
cat_cols = ["Proto", "sDSb", "dDSb", "Cause", "State", "Attack Tool"]
for col in cat_cols:
    print(f"\n{col}: {df[col].nunique()} valeurs uniques")
    print(df[col].unique()[:10])


section("9. VERIFICATION RAPIDE ANTI-LEAKAGE")
print(df.groupby("Label")[["TotBytes", "Rate", "Dur"]].mean())
# Objectif: s'assurer qu'aucune colonne (hors Attack Type / Attack Tool, deja
# exclues) ne separe les classes de facon suspecte / trop parfaite.


section("10. RESUME / PROCHAINE ETAPE")
print(
    """
Une fois les sections 6 a 9 analysees, on peut figer :
  - la liste finale des colonnes a garder / a dropper
  - contracts/schema.py (types + colonnes)
  - contracts/feature_registry.yaml

Colonnes deja actees comme a dropper :
  - Unnamed: 0        (index residuel d'un export pandas precedent)
  - Seq               (pas d'info chronologique utilisable)
  - Mean, Sum, Min, Max (redondantes avec Dur, a confirmer section 3)
  - Attack Type / Attack Tool -> a exclure des features (garder comme labels)

Decision actee pour le simulateur de replay :
  - Pas de reconstruction d'un ordre chronologique reel (impossible avec ce fichier)
  - Shuffle des lignes + emission a un rythme controle vers Kafka/Redpanda
  - Conserver les proportions reelles de classes (738k malicious / 477k benign)
  - A documenter clairement comme limitation des donnees dans le rapport final
"""
)
