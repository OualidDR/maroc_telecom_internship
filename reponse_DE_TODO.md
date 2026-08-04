# Réponses à DE_TODO.md

Salut,

Merci pour ce récap, très clair et bien priorisé. Voici mes décisions/réponses point par point, dans le même ordre.

**Contrainte à garder en tête pour toutes ces décisions :** ma machine dev fait 16GB RAM / 2 cœurs. Kafka+Spark+MinIO ensemble font déjà ramer un peu (on a des `WARN falling behind` sur les micro-batchs Spark). Chaque conteneur supplémentaire compte — c'est pour ça que certaines de mes réponses ci-dessous s'écartent de tes options initiales.

---

## 1. MLflow — ni Option A complète, ni Option B

Ta préférence pour A se justifie bien (garder le workflow de promotion sans rebuild), mais Postgres + serveur MLflow, ça fait 2 conteneurs de plus sur une machine qui est déjà limite.

**Proposition : "Option A allégée"**
- 1 conteneur serveur MLflow, backend **SQLite** (pas Postgres) — mais le fichier SQLite est géré par ce **seul** conteneur, personne d'autre n'y touche directement. Le problème de concurrence SQLite qu'on veut éviter, c'est plusieurs process qui écrivent en direct dans le fichier — pas le cas ici puisque toi (FastAPI) et moi on parlera au serveur en HTTP, jamais au fichier directement.
- Artefacts pointés vers mon MinIO existant (bucket `mlflow-artifacts`, je le crée).
- J'ajoute ça au `compose.yml` cette semaine, je te donne l'URL interne (`http://mlflow:5000`) dès que c'est up.

Dis-moi si SQLite-derrière-un-serveur te semble suffisant pour ton usage, ou s'il y a une vraie raison technique d'avoir Postgres que je rate.

## 2. Schéma `/predict/batch` — je préfère garder le preprocessing chez toi

Je préfère **ne pas** reproduire le one-hot encoding + sentinelles côté Spark — sinon cette logique existe à deux endroits (chez toi et dans Spark), et le jour où tu ajustes `feature_engineering.py`, il faudra que je pense à répercuter côté Spark aussi. Exactement le genre de déssynchronisation qu'on essaie d'éviter partout ailleurs (le contrat, les schémas générés dynamiquement...).

**Proposition :**
- Tu ajoutes `/predict/raw`, qui prend les **39 features brutes du contrat** (ce que j'ai déjà en Silver, sans transformation)
- Le preprocessing (one-hot, sentinelles, `is_tcp`/`has_dst_reply` — que j'ai déjà côté Silver d'ailleurs pour ces deux derniers) reste **entièrement de ton côté**, une seule source de vérité
- Fréquence : mes micro-batchs Spark font ~500 lignes toutes les 10s en pic, mais en pratique souvent moins (voir mes logs `[GX] batch=... rows=...`). Dimensionne plutôt pour des batchs de 50-500 lignes, pas 1000.

## 3. Contrat v1.0.1 — déjà réglé de mon côté

Bonne nouvelle : `spark_bronze_silver.py` génère son schéma Silver **dynamiquement depuis `FEATURE_SCHEMA`** (pas une liste recopiée à la main) — donc dès que `Offset` est passé dans `DROPPED_COLUMNS`, il a automatiquement arrêté d'être écrit en Silver, sans que j'aie eu besoin de toucher au code Spark.

Ta préférence (`Offset` en Bronze pour audit, absent de Silver) est **déjà exactement ce qui se passe** : Bronze garde le JSON brut tel quel (donc `Offset` y est toujours), Silver ne l'a jamais. Rien à faire.

## 4. `/drift` — périodique, pas par batch

Un appel après chaque micro-batch (~toutes les 10s) serait beaucoup trop fréquent et coûteux pour rien. Je propose plutôt :
- Un appel **périodique** (toutes les 2-4h pour commencer, à ajuster), orchestré par **Airflow** — qui est prévu dans le brief mais que je n'ai pas encore construit. C'est le bon endroit pour ce genre de tâche planifiée plutôt que du push en direct depuis Spark.
- Stockage des rapports HTML : je crée un dossier `gold/monitoring/drift_reports/` sur MinIO, tu écris dedans (ou tu me donnes le format et j'écris depuis Airflow après appel de ton endpoint — à voir ce qui est le plus simple pour toi)
- Le seuil à 50% de colonnes en drift me semble raisonnable pour démarrer, on ajustera si trop/pas assez sensible une fois qu'on a des vraies données en continu

## 5. Dockerfile FastAPI — Docker Compose, pas Kubernetes pour l'instant

- On reste sur **Docker Compose** pour le moment — Kubernetes est prévu en S5 de mon planning, pas encore construit. Pas la peine de complexifier ton Dockerfile pour du K8s qu'on n'a pas encore.
- Oui pour le readiness probe basé sur `/health` (`model_loaded: true`)
- **Pas d'auth (JWT ou autre) pour l'instant** — hors périmètre réaliste vu le temps qu'il nous reste sur le stage. On le note comme limitation/perspective dans le rapport final plutôt que de l'implémenter.

## 6. Persistance Gold — d'accord avec ta préférence

Oui, ta FastAPI reste stateless, c'est moi qui persiste : je récupère la réponse de `/predict/batch` (ou `/predict/raw` une fois qu'on l'a), je la joins à la ligne Silver correspondante, et j'écris `gold_model_predictions`/`gold_security_alerts` côté Spark — même pattern que ce que j'ai déjà fait pour Great Expectations (`foreachBatch`).

---

## Récap des actions de mon côté, dans l'ordre

1. Ajouter MLflow (serveur + SQLite, artefacts sur MinIO) au `compose.yml`
2. Te donner l'URL du tracking server
3. Créer le bucket/dossier MinIO pour les rapports de drift
4. Une fois `/predict/raw` dispo chez toi : brancher Spark dessus, puis construire la persistance Gold (point 6)
5. Airflow pour l'orchestration `/drift` (à construire, pas encore commencé)

Je regarde le point 1 (MLflow) aujourd'hui/demain comme tu le suggérais — je te tiens au courant dès que c'est up. Dispo pour le call de 15 min si besoin, sinon ce message couvre tout je pense.

À plus,
[Ton nom]
