# Guide de démarrage — 5G-NIDD (pour [Nom du collègue])

Ce guide te permet de faire tourner le projet en local, de A à Z. Suis les étapes dans l'ordre — chaque section indique si elle est **obligatoire** ou **optionnelle** selon ce que tu veux faire (juste ton pipeline ML vs. le pipeline complet).

---

## 0. Prérequis

| Outil | Pourquoi | Vérifier avec |
|---|---|---|
| Git | cloner le repo | `git --version` |
| Python 3.12 | tout le code | `python3 --version` |
| Docker + Docker Compose | Kafka/Redpanda + MinIO (uniquement si tu testes le pipeline DE) | `docker --version` |
| Java 17 (JDK) | Spark en a besoin (uniquement si tu testes Spark) | `java -version` |

Si Java manque :
```bash
sudo apt install openjdk-17-jdk-headless
```

---

## 1. Cloner et configurer l'environnement Python (obligatoire)

```bash
git clone https://github.com/OualidDR/maroc_telecom_internship.git
cd maroc_telecom_internship

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Config locale (obligatoire)

```bash
cp .env.example .env
```
Ouvre `.env` et remplis au minimum, si tu as reçu tes accès :
```
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...          # ton compte à toi, pas celui du DE
SNOWFLAKE_PASSWORD=...
```
Le reste (`KAFKA_BOOTSTRAP_SERVERS`, `MINIO_*`) n'est utile que si tu lances aussi l'infra locale (section 4).

## 3. Le dataset (obligatoire si tu entraînes depuis le CSV)

```bash
mkdir -p data/raw
```
Télécharge `Combined.csv` depuis Kaggle :
`https://www.kaggle.com/datasets/tinasheaustin/5g-nidd-updated-data-august-2025`
→ place-le dans `data/raw/Combined.csv`

**À ce stade, tu peux déjà lancer ton propre pipeline** (`modeling/src/preprocessing.py`, `splits.py`, `train.py`...) sans rien d'autre — il lit le CSV directement et applique `contracts/schemas.py`. Les sections suivantes ne sont nécessaires que si tu veux tester/observer le pipeline Kafka→Spark→Snowflake→dbt.

---

## 4. Infra locale : Kafka/Redpanda + MinIO (optionnel)

```bash
docker compose up -d
docker ps
```
Tu dois voir `redpanda`, `redpanda-console`, `minio` actifs (`minio-init` s'arrête après avoir créé les buckets, `Exited (0)` est normal).

Crée le topic Kafka (une seule fois) :
```bash
docker exec -it redpanda rpk topic create flows-raw --partitions 3
```

**Interfaces web :**
- Redpanda Console : http://localhost:8080
- MinIO Console : http://localhost:9001 (identifiants dans ton `.env`)

## 5. Faire tourner le pipeline de streaming complet (optionnel)

Repart toujours d'un état propre avant un test :
```bash
chmod +x reset_pipeline.sh
./reset_pipeline.sh
```

Puis, dans deux terminaux séparés :
```bash
# Terminal 1 -- attends "Streaming started..." avant de continuer
python3 streaming/spark_bronze_silver.py
```
```bash
# Terminal 2
python3 ingestion/replay_simulator.py --rate 200 --limit 2000 --seed 7
```
Dans le Terminal 1, tu dois voir apparaître `[GX] batch=... -> OK` toutes les ~10s — c'est Great Expectations qui valide chaque micro-batch en direct.

## 6. Snowflake (optionnel, nécessite tes accès en lecture seule)

Si tu n'as pas encore d'accès, demande au DE — pas besoin de créer toi-même les tables/warehouse.

Charger les données (le DE l'a peut-être déjà fait, à vérifier avant de dupliquer) :
```bash
python3 warehouse/load_silver_to_snowflake.py
python3 warehouse/load_quality_log_to_snowflake.py
```

## 7. dbt (optionnel)

```bash
pip install dbt-snowflake
mkdir -p ~/.dbt
```
Crée `~/.dbt/profiles.yml` avec ce contenu (utilise le template `profiles.yml.example` du repo) :
```yaml
nidd_dw:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
      user: "{{ env_var('SNOWFLAKE_USER') }}"
      password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
      role: SYSADMIN
      database: "{{ env_var('SNOWFLAKE_DATABASE', 'NIDD_DB') }}"
      warehouse: "{{ env_var('SNOWFLAKE_WAREHOUSE', 'NIDD_WH') }}"
      schema: DBT_DEV
      threads: 2
```
Puis :
```bash
set -a; source .env; set +a
cd warehouse/dbt/nidd_dw
dbt debug     # doit afficher "All checks passed!"
dbt seed
dbt run
dbt test
```

---

## Où trouver quoi dans le repo

```
contracts/schemas.py          # LE contrat de données -- lis ça en premier
modeling/                     # ton pipeline à toi (preprocessing, train, etc.)
ingestion/replay_simulator.py # CSV -> Kafka
streaming/spark_bronze_silver.py  # Kafka -> Bronze/Silver (MinIO) + validation GX
quality/                      # Great Expectations
warehouse/                    # scripts de chargement Snowflake + dbt
```

---

## Dépannage — problèmes déjà rencontrés

| Symptôme | Cause | Solution |
|---|---|---|
| `Could not resolve host: github.com` | coupure réseau temporaire | relancer la commande |
| `Invalid username or token` (git push) | mot de passe classique refusé par GitHub | utiliser un Personal Access Token, ou passer en SSH |
| `mc: <ERROR> Access Denied` | mauvais conteneur/alias `mc` | utiliser `docker compose run --rm --entrypoint sh minio-init -c "..."` |
| `File data/raw/Combined.csv is 262MB, exceeds 100MB` (git push) | CSV pas dans `.gitignore` avant le premier commit | ne jamais commiter `data/`, déjà dans `.gitignore` |
| `ModuleNotFoundError: No module named 'distutils'` (Spark) | incompatibilité PySpark 3.5.1 / Python 3.12 | déjà corrigé dans `gx_spark_validator.py` (pas de `toPandas()`) |
| `DataContextError: datasource already exists` (GX) | contexte GX persistant sur disque | déjà corrigé (contexte éphémère dans le validateur Spark) |
| `DELTA_INVALID_CHARACTERS_IN_COLUMN_NAMES` | espaces dans les noms de colonnes | déjà corrigé (`attack_type`/`attack_tool` renommés) |
| `invalid identifier` (Snowflake) | casse des colonnes (Snowflake stocke en MAJUSCULES) | les scripts de chargement mettent déjà tout en majuscules avant l'envoi |
| `Insufficient privileges... CREATE SCHEMA` (dbt) | rôle Snowflake sans droits sur la database | demander au DE de lancer les `GRANT` nécessaires |
| Warning `KAFKA-1894` en boucle dans les logs Spark | comportement interne connu du client Kafka | inoffensif, ignorer |
| `docker ps` ne montre pas `minio-init` | conteneur "one-shot", se termine après son travail | normal, vérifier avec `docker ps -a` |

Si tu tombes sur autre chose, demande au DE avant de tout réinstaller — il y a de bonnes chances que ce soit déjà résolu quelque part dans l'historique du projet.
