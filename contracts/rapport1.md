# Contrat de données — 5G-NIDD

**De :** [Ton nom] (Data Engineer)
**Pour :** [Nom du collègue] (Data Scientist)
**Date :** Juillet 2026
**Fichier de référence :** `contracts/schemas.py`

---

## Objectif de ce document

Ce rapport résume l'exploration du dataset brut `Combined.csv` et les décisions prises sur les colonnes à garder, dropper, ou traiter comme labels. Ces décisions sont maintenant figées dans `contracts/schemas.py`, qui sert de **source unique de vérité** pour nos deux pipelines (DE et DS). Merci d'importer directement depuis ce fichier plutôt que de recopier des listes de colonnes dans ton code — ça évite qu'on désynchronise sans s'en rendre compte.

```python
from contracts.schemas import FEATURE_COLUMNS, LABEL_COLUMNS, DROPPED_COLUMNS
```

---

## 1. Ce qu'on a constaté sur les données brutes

- **1 215 890 lignes**, **52 colonnes** — conforme au brief.
- **Label binaire** : 738 153 Malicious / 477 737 Benign.
- **Attack Type** (9 classes) : Benign, UDPFlood, HTTPFlood, SlowrateDoS, TCPConnectScan, SYNScan, UDPScan, SYNFlood, ICMPFlood — répartition très déséquilibrée (UDPFlood = 457k lignes, ICMPFlood = 1 155 lignes seulement). À anticiper dans ton choix de métriques/stratégie d'entraînement.
- **Attack Tool** (6 valeurs, non documentée dans le brief initial) : indique quel outil a généré chaque flux (Hping3, Goldeneye, Nmap, Slowloris, Torshammer). Fortement corrélée au label → **à exclure des features**, même logique que `Attack Type`.
- **Aucune colonne IP/port** dans cette version du fichier (`SrcAddr`, `DstAddr`, etc. absentes) — donc pas de risque de leakage par adresse, contrairement à ce que supposait le brief initial.
- **0 doublon** exact sur l'ensemble du dataset.

## 2. Colonnes supprimées (9) et pourquoi

| Colonne | Raison |
|---|---|
| `Unnamed: 0` | Index résiduel d'un export pandas précédent, pas une vraie feature |
| `Seq` | Pas un identifiant chronologique fiable : se réinitialise ~128 379 fois sur 1,2M lignes (bookkeeping interne Argus), sans lien avec les sessions d'attaque |
| `RunTime`, `Mean`, `Sum`, `Min`, `Max` | 100% identiques à `Dur` sur toutes les lignes — redondantes |
| `sVid`, `dVid` | Variance nulle (valeur constante = 610 quand non nulle) + >90% de valeurs manquantes — aucun signal |

## 3. Pas de signal temporel exploitable — impact sur le simulateur (et donc sur tes données d'entraînement)

On a vérifié si l'ordre des lignes du CSV correspondait à un ordre chronologique réel (ce qui aurait permis de faire un vrai replay temps réel) :
- `Seq` ne fonctionne pas (voir ci-dessus).
- Les blocs contigus de `Attack Type` dans le fichier sont **21 121 blocs**, taille médiane de seulement 2 lignes → les types d'attaques sont éparpillés dans tout le fichier, pas organisés en grandes sessions.

**Conclusion :** impossible de reconstituer l'ordre chronologique réel des événements à partir de ce fichier. Le simulateur de replay va donc **mélanger aléatoirement les lignes** (proportions de classes conservées) plutôt que de rejouer le fichier dans son ordre d'origine. À mentionner dans la partie "limites des données" du rapport final commun — ce n'est pas un problème de notre pipeline, c'est une limite du fichier source.

## 4. Valeurs manquantes — structurelles, pas aléatoires

Deux grandes familles de colonnes ont des NaN, et dans les deux cas c'est **normal, pas une erreur de collecte** :

- **Colonnes `d*`** (`dTos`, `dTtl`, `dHops`, `dDSb`) : NaN dans ~77% des lignes → nulles quand la destination ne répond jamais (ex. flux UDP à sens unique, scans).
- **Colonnes spécifiques TCP** (`SrcWin`, `DstWin`, `SrcTCPBase`, `DstTCPBase`, `SrcGap`, `DstGap`) : NaN pour tout ce qui n'est pas `tcp` (udp, icmp, arp...).

**Recommandation pour ton feature engineering :** ne pas imputer par moyenne/médiane, ça mélangerait des flux qui n'ont simplement pas ce type de donnée avec des flux où la valeur serait vraiment à 0. On a prévu deux colonnes indicatrices dans le contrat :
- `is_tcp` (1 si `Proto == 'tcp'`)
- `has_dst_reply` (1 si `dTtl` non nul)

Utilise-les en complément d'une valeur sentinelle (ex. -1) sur les colonnes concernées, pour que le modèle distingue "non applicable" de "applicable mais nul".

## 5. Point d'attention sur `SrcTCPBase` / `DstTCPBase`

Ces colonnes montent jusqu'à ~4,29 milliards. Vérifié : ce n'est **pas une valeur sentinelle d'erreur**, c'est `2³² - 1`, le maximum légitime d'un numéro de séquence TCP initial (ISN) sur 32 bits. Donnée correcte, mais à forte cardinalité et quasi aléatoire — probablement peu utile telle quelle comme feature brute ; à toi de voir si tu veux la transformer (ex. buckets, ou la dropper côté modèle) plutôt que de la garder telle quelle.

## 6. Schéma final

- **40 colonnes features** (types + nullabilité documentés dans `FEATURE_SCHEMA`)
- **3 colonnes labels** : `Label` (binaire), `Attack Type` (multiclasse), `Attack Tool` (métadonnée à exclure de l'entraînement)
- **9 colonnes droppées** (liste ci-dessus)
- Total : 40 + 3 + 9 = 52 ✅ (correspond bien aux colonnes du CSV brut)

## 7. Format des événements Kafka (pour information, pas encore livré)

Chaque flux sera emballé ainsi avant d'être poussé sur Kafka/Redpanda :
```json
{
  "event_id": "evt_00000123",
  "ingestion_timestamp": "2026-07-09T12:00:00.000Z",
  "schema_version": "1.0.0",
  "flow": { "...40 features + 3 labels..." }
}
```

---

## Prochaines étapes de mon côté

- Simulateur de replay (CSV → Kafka/Redpanda)
- Jobs Spark Bronze/Silver

Dis-moi si tu vois un problème avec une des décisions ci-dessus, ou si tu as besoin d'une colonne que j'ai droppée pour ton feature engineering — c'est le bon moment pour en discuter avant que le pipeline soit branché dessus.
