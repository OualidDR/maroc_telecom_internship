# Prochaine étape côté DE : MLflow tracking server dans le compose

## Contexte rapide

De mon côté c'est fait : le code lit maintenant le tracking URI depuis une variable d'environnement `MLFLOW_TRACKING_URI` (avec fallback sur `http://localhost:5000`). Toutes les scripts, la FastAPI et les notebooks utilisent ce pattern.

Ce qui manque : le serveur MLflow lui-même, en container, backé par SQLite sur un volume Docker + tes buckets MinIO existants pour les artefacts.

---

## Ce que tu ajoutes au `compose.yml`

Un nouveau service `mlflow` :

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.16.0
  container_name: mlflow
  ports:
    - "5000:5000"
  volumes:
    - mlflow_data:/mlflow
  environment:
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    - AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY}
    - AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY}
  command: >
    mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination s3://mlflow-artifacts
      --serve-artifacts
  depends_on:
    - minio
```

Et le volume à déclarer en bas du fichier :

```yaml
volumes:
  mlflow_data:
```

---

## Deux points à ne pas oublier

1. **Créer le bucket `mlflow-artifacts` sur MinIO.** Soit manuellement via la console MinIO (`http://localhost:9001`), soit ajoute la création au job `minio-init` que tu as déjà.

2. **Les vars d'env `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`.** Elles doivent être définies dans ton `.env` — normalement c'est déjà le cas si tu utilises déjà MinIO. Sinon ping-moi.

---

## Comment vérifier que ça marche

Une fois lancé :

```powershell
docker compose up -d mlflow
docker logs mlflow
```

Puis dans un navigateur : `http://localhost:5000` doit afficher l'UI MLflow (vide pour le moment).

Ping-moi quand c'est up. De mon côté je re-registre le modèle sur ce nouveau serveur en une commande, puis on teste que la FastAPI le charge correctement.

---

## Ce qui vient après (pour ta planif)

Une fois MLflow up et le modèle re-registré :

1. J'écris le Dockerfile de la FastAPI
2. Tu l'ajoutes au `compose.yml`
3. On teste le flux end-to-end : Spark → `/predict/batch` → prédictions → écriture Gold

Merci !
