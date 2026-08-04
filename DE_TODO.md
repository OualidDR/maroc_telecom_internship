# Ce dont j'ai besoin de ton côté (DE) pour avancer

Salut,

J'ai fini tout le côté DS de bout en bout (modèle, MLflow registry, FastAPI avec `/predict`, `/predict/batch`, `/drift`, monitoring Evidently). Pour brancher tout ça sur ton pipeline et finaliser la partie déploiement, j'ai besoin de quelques décisions et setup de ton côté. Voici la liste par ordre de priorité.

---

## 1. Décision architecture MLflow (bloquant pour Docker)

**Contexte :** actuellement MLflow tourne en local avec `sqlite:///mlflow.db` + un dossier `mlruns/` sur ma machine. Ça marche pour dev solo, mais ça ne passe pas en conteneur — un fichier SQLite ne se partage pas entre containers, et si on scale la FastAPI à plusieurs replicas ils vont se marcher dessus.

**Ce qu'il faut décider :**
- **Option A (production-grade) :** MLflow tracking server en container, backé par Postgres pour les métadonnées + MinIO pour les artefacts.
- **Option B (simple) :** on fige le modèle dans l'image Docker de la FastAPI, pas de registry runtime.

**Ma préférence : Option A**, parce que ça préserve le workflow "promouvoir v2 → réaffecter l'alias staging" sans rebuild du container. Et tu as déjà MinIO dans ton `compose.yml`, donc l'infra est à moitié en place.

**Ce que je te demande :**
- Confirme qu'on part sur A ou B
- Si A : ajoute au `compose.yml` un service MLflow tracking server + un service Postgres, en pointant MLflow vers ton MinIO existant pour les artefacts
- Donne-moi l'URL du tracking server une fois up (probablement quelque chose comme `http://mlflow:5000` en réseau Docker interne)

De mon côté après ça : une ligne à changer dans `api.py` (env variable `MLFLOW_TRACKING_URI`), ré-enregistrer le modèle sur le nouveau serveur, tester en local.

---

## 2. Schéma du payload que Spark enverra à `/predict/batch`

**Contexte :** j'ai un endpoint `POST /predict/batch` qui accepte jusqu'à 1000 flows par requête. Il attend un JSON de cette forme :

```json
{
  "flows": [
    {"Proto_udp": 1.0, "Dur": 2.57, "TotBytes": 84.0, "sHops": 1.0, ...},
    {"Proto_udp": 0.0, "Dur": 5.14, "TotBytes": 1200.0, "sHops": 3.0, ...}
  ]
}
```

Chaque flow doit contenir les **76 features** dans la forme post-préprocessing (one-hot déjà appliqué, sentinelles `-1` pour les NaN, indicateurs `is_tcp` et `has_dst_reply` calculés).

**Ce que je te demande :**
- Confirme que ton job Spark peut produire ce schéma exact depuis la Silver layer, ou dis-moi si tu préfères que je fasse le préprocessing côté API
- Si tu veux que je préprocesse côté API, il faut qu'on ajoute un endpoint `/predict/raw` qui prend les features brutes (52 colonnes du contrat) et fait le preprocessing avant scoring
- Donne-moi la fréquence attendue (batches par minute, taille moyenne d'un batch) pour que je puisse dimensionner correctement

La liste exacte des 76 features est dans `modeling/artifacts/splits/X_train.parquet` (colonnes) — je peux te l'envoyer en `.txt` si ça t'aide.

---

## 3. Contrat de données v1.0.1 — synchronisation

**Contexte :** j'ai bumpé `contracts/schemas.py` en v1.0.1 avec un changement : `Offset` a été retiré de `FEATURE_SCHEMA` et mis dans `DROPPED_COLUMNS`. Raison : c'est du leakage de position dans le CSV (chaque classe d'attaque occupe une plage d'offset distincte parce que le dataset a été assemblé en concaténant des captures par classe). Ça n'a aucun signal réel en streaming. Détails complets dans `modeling/notes.md`.

**Ce que je te demande :**
- Vérifie ton `spark_bronze_silver.py` — est-ce qu'il écrit encore `Offset` en Silver ?
- Deux options : soit tu le laisses en Silver pour audit mais tu ne le passes pas au modèle, soit tu l'exclus complètement en Silver
- Ma préférence : le laisser en Bronze (audit trail), l'exclure en Silver (ce que voit le modèle)

---

## 4. Endpoint `/drift` — comment tu veux l'appeler ?

**Contexte :** j'ai un endpoint `POST /drift` qui prend un batch de flows et retourne un résumé Evidently avec les colonnes en drift + génère un rapport HTML. Le dataset de référence est construit à partir de mon jeu d'entraînement.

**Ce que je te demande :**
- Est-ce que tu veux l'appeler depuis un DAG Airflow toutes les X heures, ou en push depuis Spark après chaque batch de N flows ?
- Où on stocke les rapports HTML générés ? (Actuellement dans `modeling/artifacts/monitoring/reports/`, faudra probablement les envoyer sur MinIO)
- Quel seuil de drift déclenche une alerte selon toi ? Actuellement Evidently utilise 50% des colonnes en drift, ce qui me paraît raisonnable

---

## 5. Dockerfile pour la FastAPI

**Contexte :** une fois qu'on a réglé MLflow (point 1), je peux écrire un Dockerfile pour ma FastAPI. C'est mon boulot mais j'ai besoin de savoir quel type de deployment tu vises.

**Ce que je te demande :**
- Confirme la structure de compose ou k8s que tu vises pour l'inférence
- Est-ce qu'il faut un `readiness probe` ? (mon endpoint `/health` peut servir à ça, il retourne `model_loaded: true` quand le modèle est chargé)
- Est-ce que tu veux que je gère l'auth (JWT ?) ou tu mets un reverse proxy devant qui s'en occupe ?

---

## 6. Persistance des prédictions et alertes (Gold layer)

**Contexte :** l'archi mentionne une Gold layer avec `gold_model_predictions`, `gold_security_alerts`. Actuellement mon API retourne juste la prédiction dans la response HTTP — rien n'est persisté côté service.

**Ce que je te demande :**
- C'est toi qui persistes en Gold côté Spark (tu prends la response de `/predict/batch` et tu la joins à la Silver row correspondante avant écriture Gold), ou tu veux que ma FastAPI écrive directement dans MinIO / Snowflake ?
- Ma préférence : c'est ton côté, ma FastAPI reste stateless. Ça sépare bien les responsabilités.

---

## Ce que j'ai déjà fait pour info

Pour que tu situes où on en est côté DS :

- **Modeling :** XGBoost multiclass (9 classes), macro-F1 0.92 sur validation. LogReg + RF comparés aussi
- **Contract updates :** v1.0.1 avec drop de `Offset` (raison documentée)
- **MLflow :** tracking + registry local, modèle enregistré comme `5g-nidd-attack-classifier@staging` (v1)
- **SHAP :** explications globales + locales dispos via `?explain=true` sur les endpoints predict
- **FastAPI :** `/health`, `/model/info`, `/predict`, `/predict/batch`, `/drift` — tous testés
- **Evidently :** dataset de référence buildé, détection validée dans les deux sens (batch clean → pas de drift, batch shifté artificiellement → drift détecté)
- **Docs :** toutes les décisions et les ablations dans `modeling/notes.md`

Tout ça est sur la branche `main` (mergé depuis `ds/eda-baseline`, `ds/serving-and-monitoring`, `ds/drift-monitoring`).

---

## Ordre suggéré pour toi

1. **Aujourd'hui / demain :** décision Option A vs B pour MLflow (point 1)
2. **Cette semaine :** setup MLflow server + confirmation schéma payload (points 1 + 2)
3. **Ensuite :** contrat v1.0.1 + drift orchestration (points 3 + 4)
4. **Enfin :** deployment + gold layer (points 5 + 6)

Le point 1 est vraiment le blocage principal — tant qu'on n'a pas décidé, je ne peux pas Dockeriser proprement, et donc je ne peux pas non plus t'aider à tester l'intégration end-to-end.

Ping-moi quand tu as regardé, on peut faire un call de 15 min si c'est plus simple.

Merci !
