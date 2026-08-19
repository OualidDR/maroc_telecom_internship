# Run complet du pipeline — 150 000 lignes

Suivre les étapes dans l'ordre. Chaque bloc de commandes est indépendant, à copier-coller tel quel.

---

## 0. Prérequis — TOUS les conteneurs up

`reset_pipeline.sh` (étape 1) fait `docker exec redpanda ...` et les étapes 2-3
ont besoin de Kafka → Redpanda doit tourner. Le scoring (étape 5) passe par la
vraie API (`PREDICT_API_MOCK=false`) → MLflow + api doivent tourner aussi. Le
plus simple : tout démarrer.

```bash
docker compose up -d          # redpanda, minio, mlflow, api (+ console, minio-init)
docker ps                     # redpanda, minio, mlflow, nidd-api doivent être "Up" (api "healthy")
docker logs minio-init        # doit finir par "buckets ready"
curl http://localhost:8000/health    # {"status":"ok","model_loaded":true}
```

> Attendre que `nidd-api` soit **healthy** (le chargement du modèle depuis
> MLflow prend ~30-60s au démarrage) avant l'étape 5.

---

## 1. Reset complet

```bash
./reset_pipeline.sh
```

Nettoie Bronze/Silver/Gold + checkpoints + quality_log sur MinIO (pas
`mlflow-artifacts` : le modèle survit au reset).

---

## 2. Terminal 1 — Spark (Bronze/Silver + Great Expectations)

```bash
python3 streaming/spark_bronze_silver.py
```

Attendre de voir :
```
SparkBatchValidator ready -- GX suite built once, reused per micro-batch.
Streaming started. Bronze -> s3a://bronze/flows  Silver -> s3a://silver/flows
Press Ctrl+C to stop.
```

---

## 3. Terminal 2 — Simulateur (150k lignes, en arrière-plan)

```bash
nohup python3 ingestion/replay_simulator.py --rate 300 --limit 150000 --seed 7 --spread-hours 48 > simulator_150k.log 2>&1 &
tail -f simulator_150k.log
``` 

Attendre `Done. Sent 150000 events...` dans le log (Ctrl+C arrête juste le `tail`, pas le process).

`--spread-hours 48` : répartit les `ingestion_timestamp` sur les dernières 48h
(synthétique) au lieu de tout marquer « maintenant », sinon les visuels horaires
Power BI seraient un seul point. Sans cette option, tout tombe dans un seul
bucket d'heure.

---

## 4. Vérifier que Spark a rattrapé le retard

Le simulateur finit d'envoyer en ~8 min, mais Spark draine ensuite à ~500
lignes/s (`MAX_OFFSETS_PER_TRIGGER=5000`), donc ~5 min pour vider 150k.
Retourner au Terminal 1, attendre ~30s sans nouveau `[GX] batch=...` (= tout
est drainé), puis arrêter Spark proprement :

```
Ctrl+C
```

---

## 5. Job de prédiction (mode RÉEL via l'API) → Gold

Nécessite l'API + MLflow up (étape 0). Le client découpe le batch en requêtes
de 500 lignes (`PREDICT_API_MAX_ROWS`), donc 150k = ~300 appels, quelques
minutes. Surveiller la RAM (MLflow 3g + api 1g + redpanda + minio).

```bash
python3 serving/spark_predict_and_gold.py
```

Attendre `[predict_and_gold] batch=0 scored=150000 alerts=...` (le `scored`
doit être ~150000 ; s'il manque des lignes, voir les WARN `[predict_client]`),
puis :

```
Ctrl+C
```

---

## 6. Charger vers Snowflake (reset complet des tables)

**Une seule fois** (si pas déjà fait) — créer les tables Gold avant le premier
chargement. Le loader utilise `auto_create_table=False`, donc les tables doivent
exister. Ouvrir un worksheet Snowflake et exécuter :

```sql
-- contenu de warehouse/create_gold_predictions.sql
-- (crée GOLD_MODEL_PREDICTIONS + GOLD_SECURITY_ALERTS, avec flow_timestamp)
```

Puis charger (à chaque run) :

```bash
python3 warehouse/load_silver_to_snowflake.py --truncate
python3 warehouse/load_quality_log_to_snowflake.py --truncate
python3 warehouse/load_gold_predictions_to_snowflake.py --truncate
```

> **Power BI — axe temporel des alertes :** dans `GOLD_SECURITY_ALERTS` /
> `GOLD_MODEL_PREDICTIONS`, utiliser **`FLOW_TIMESTAMP`** (le moment du flux, =
> la timeline SOC synthétique) pour les visuels « alertes dans le temps », PAS
> `PREDICTION_TIMESTAMP` (qui vaut ~l'heure du scoring, quasi identique pour
> toutes les lignes). Les pages trafic (Silver) utilisent déjà
> `INGESTION_TIMESTAMP`, cohérent avec `FLOW_TIMESTAMP`.

---

## 7. Recalculer les marts dbt

```bash
cd warehouse/dbt/nidd_dw
dbt run
dbt test
cd ../../..
```

---

## 8. Vérification finale (Snowflake, worksheet)

```sql
SELECT COUNT(*) FROM SILVER.SILVER_FLOWS;
SELECT COUNT(*) FROM DBT_DEV.MART_ATTACK_SUMMARY;
SELECT ATTACK_TYPE, SUM(FLOW_COUNT) FROM DBT_DEV.MART_ATTACK_SUMMARY GROUP BY ATTACK_TYPE ORDER BY 2 DESC;
```
