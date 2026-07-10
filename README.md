# 5G-NIDD -- Détection d'intrusion réseau 5G en temps réel

Stage d'été 2A -- ENSIAS. Équipe : 1 Data Engineer + 1 Data Scientist.

## Démarrage rapide (à faire sur CHAQUE machine, DE et DS)

```bash
git clone https://github.com/OualidDR/maroc_telecom_internship.git
cd maroc_telecom_internship

# 1. Environnement Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Config locale (jamais commitée)
cp .env.example .env

# 3. Dataset (trop volumineux pour Git, à télécharger séparément)
mkdir -p data/raw
# Télécharger Combined.csv depuis Kaggle:
# https://www.kaggle.com/datasets/tinasheaustin/5g-nidd-updated-data-august-2025
# -> placer le fichier dans data/raw/Combined.csv

# 4. Infra locale (Kafka/Redpanda + MinIO)
docker compose up -d
docker ps   # confirme que redpanda, minio, redpanda-console tournent
docker logs minio-init   # doit finir par "buckets ready"

# 5. Créer le topic Kafka (une seule fois)
docker exec -it redpanda rpk topic create flows-raw --partitions 3
```

⚠️ **Chacun fait tourner sa PROPRE infra locale** (Redpanda + MinIO ne sont pas partagés entre les deux machines). Ce qui est partagé, c'est le **code** via Git -- pas les conteneurs Docker. Voir la section "Pourquoi pas d'infra partagée" plus bas.

## Structure du repo

```
contracts/
  schemas.py              # Contrat de données -- SOURCE UNIQUE DE VÉRITÉ
                           # (colonnes gardées/droppées/labels, types, nullabilité)
ingestion/
  replay_simulator.py      # CSV -> Kafka/Redpanda (shuffle, pas d'ordre chronologique)
streaming/
  spark_bronze_silver.py   # Kafka -> Bronze (brut) -> Silver (typé) sur MinIO
data/
  raw/Combined.csv         # gitignored -- à télécharger, voir étape 3 ci-dessus
docker-compose.yml         # Redpanda + MinIO, sized pour laptop 16GB/2 coeurs
.env.example                # template de config -- copier vers .env
```

## Workflow Git quotidien

```bash
git pull                          # récupérer les derniers changements
# ... travailler ...
git add .
git commit -m "description du changement"
git push
```

Prévenez-vous mutuellement sur Slack/WhatsApp après un push qui touche
`contracts/schemas.py` -- c'est le fichier partagé le plus sensible, un
changement dessus impacte directement l'autre pipeline.

## Tester le pipeline d'ingestion

```bash
# Terminal 1 -- Spark (consomme Kafka, écrit sur MinIO)
python3 streaming/spark_bronze_silver.py

# Terminal 2 -- simulateur (CSV -> Kafka)
python3 ingestion/replay_simulator.py --rate 200 --limit 2000
```

Vérifier dans la console MinIO (http://localhost:9001, identifiants dans
`.env`) que des fichiers Delta apparaissent dans `bronze/flows` et
`silver/flows`.

## Pourquoi pas d'infra partagée (Redpanda/MinIO communs) ?

`docker compose up` tourne en local sur chaque laptop -- `localhost:9000`
sur la machine du DE ne veut rien dire pour le DS, et inversement. Pour
l'instant chacun développe avec sa propre instance locale, et seul le
**code** (contrats, jobs Spark, scripts) est partagé via Git. Si le
projet a besoin de données réellement partagées entre les deux (ex: le DS
doit lire les données Silver produites par le DE), il faudra héberger
Redpanda/MinIO sur une machine accessible aux deux (VM, serveur de
l'école) -- à voir avec l'encadrante si besoin.

## Contrat de données

Voir `contracts/schemas.py` et `rapport_contrat_donnees_5G-NIDD.md` pour
le détail complet de l'exploration (colonnes droppées, gestion des NaN
structurels, absence de signal temporel dans le CSV brut, etc.).