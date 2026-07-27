# Power BI — état des lieux : tables disponibles et ce qu'il reste à construire

**Objectif de ce document :** savoir exactement sur quoi on peut déjà brancher Power BI, et ce qui manque avant de pouvoir construire les 4 pages prévues au brief.

---

## 1. Tables disponibles aujourd'hui dans Snowflake

| Table / Vue | Type | Contenu | Schéma |
|---|---|---|---|
| `SILVER_FLOWS` | Table brute | 1 ligne = 1 flux réseau, toutes les features du contrat | `SILVER` |
| `QUALITY_LOG` | Table brute | 1 ligne = 1 micro-batch validé par Great Expectations | `SILVER` |
| `stg_flows` | Vue (dbt staging) | Version nettoyée/renommée de `SILVER_FLOWS` | `DBT_DEV` |
| `stg_quality_log` | Vue (dbt staging) | Version nettoyée de `QUALITY_LOG` | `DBT_DEV` |
| `attack_type_category` | Seed dbt | Mapping `attack_type` → `DoS` / `Reconnaissance` / `Benign` | `DBT_DEV` |
| **`mart_attack_summary`** | **Table Gold** | Volume par `attack_type`, `attack_category`, `attack_tool`, `label` | `DBT_DEV` |
| **`mart_traffic_by_hour`** | **Table Gold** | Volume par heure, croisé `label` × `attack_type` × `attack_category` | `DBT_DEV` |
| **`mart_protocol_breakdown`** | **Table Gold** | `Proto` croisé avec `attack_type` | `DBT_DEV` |
| **`mart_quality_summary`** | **Table Gold** | Taux de succès Great Expectations par heure | `DBT_DEV` |

**Power BI doit se connecter uniquement aux 4 tables en gras** (les marts Gold) — jamais directement à `SILVER_FLOWS` ou aux vues staging, conformément au brief ("Power BI reads exclusively from Gold").

---

## 2. Couverture par page Power BI (selon le brief)

### Page 1 — SOC Overview
| Visuel prévu | Table source | Statut |
|---|---|---|
| Flux totaux (24h) | `mart_traffic_by_hour` | ✅ prêt |
| Taux de malveillance dans le temps | `mart_traffic_by_hour` | ✅ prêt |
| Répartition par attaque (dernière heure) | `mart_attack_summary` ou `mart_traffic_by_hour` | ✅ prêt |
| Alertes ouvertes | `gold_alerts` | ❌ **manquant — côté DS** |
| 10 dernières alertes critiques | `gold_alerts` | ❌ **manquant — côté DS** |

### Page 2 — Attack Analysis
| Visuel prévu | Table source | Statut |
|---|---|---|
| Attaques dans le temps | `mart_traffic_by_hour` | ✅ prêt |
| DoS vs Reconnaissance | `mart_attack_summary` / `mart_traffic_by_hour` (colonne `attack_category`) | ✅ prêt |
| Heatmap par heure/protocole | `mart_protocol_breakdown` + `mart_traffic_by_hour` | ✅ prêt (à combiner dans Power BI) |
| Top destinations (IP hashées) | `gold_top_talkers` | ❌ **manquant — pas d'IP dans le dataset actuel, à discuter** |

### Page 3 — Explainability
| Visuel prévu | Table source | Statut |
|---|---|---|
| SHAP global | export SHAP interrogeable | ❌ **manquant — côté DS** |
| Top features par attaque | export SHAP interrogeable | ❌ **manquant — côté DS** |
| Drill-down par alerte | `gold_alerts` + SHAP | ❌ **manquant — côté DS** |

### Page 4 — Model & Data Quality
| Visuel prévu | Table source | Statut |
|---|---|---|
| Taux de succès qualité dans le temps | `mart_quality_summary` | ✅ prêt |
| Enregistrements rejetés | `mart_quality_summary` (colonnes `failed_batches`, `total_failed_expectations`) | ✅ prêt |
| Version du modèle actif | `gold_model_performance` (ou export MLflow) | ❌ **manquant — côté DS** |
| Métriques offline (précision/rappel) | `gold_model_performance` | ❌ **manquant — côté DS** |

---

## 3. Résumé — ce qui reste à construire

### Déjà prêt (on peut commencer à builder dans Power BI maintenant)
- Page 1 : partiellement (tout sauf les alertes)
- Page 2 : quasi complète (sauf top talkers, à discuter — voir note ci-dessous)
- Page 4 : partiellement (tout sauf les métriques modèle)

### Manquant côté DE
- `gold_top_talkers` — **à discuter** : le dataset actuel n'a pas d'adresses IP (colonnes droppées dès l'exploration initiale, absentes du CSV source dans notre version). Soit on laisse tomber ce visuel, soit on trouve un proxy (ex: grouper par `Offset`... non, `Offset` a été droppé pour leakage. À voir s'il existe un autre identifiant réseau exploitable, sinon ce visuel sort du périmètre réalisable.)

### Manquant côté DS (voir `snowflake_integration_ds.md` et `statut_projet_et_prochaines_etapes.md` pour le détail)
- `gold_alerts` (1 ligne par prédiction)
- `gold_model_performance` (précision/rappel par version de modèle)
- Export SHAP sous forme de table interrogeable (pas des `.png`)

---

## Prochaine action concrète
On peut commencer à construire les Pages 1, 2 et 4 dans Power BI **dès maintenant** avec ce qui est prêt, et brancher les visuels manquants au fur et à mesure que le DS livre ses tables. Pas besoin d'attendre que tout soit fini pour démarrer.
