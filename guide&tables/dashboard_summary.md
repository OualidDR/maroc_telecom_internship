# 5G-NIDD SOC Dashboard — Summary & Talking Points

Reference for the Power BI dashboard: what each page shows, the key findings, and
the honest caveats to mention in the report / presentation.

---

## 1. What the dashboard is

A 4-page **Security Operations Center (SOC)** dashboard for 5G network intrusion
detection, built on the full data pipeline:

```
Combined.csv → Kafka replay → Spark Bronze/Silver (+ Great Expectations)
   → real XGBoost predictions via FastAPI → Gold layer → Snowflake → dbt marts → Power BI
```

- **Volume shown:** 150,000 network flows, 9 classes (Benign + 8 attack types).
- **Model:** XGBoost multiclass, served from MLflow registry (`v1`, alias `staging`).
- **Power BI reads Gold only** (dbt marts + Gold prediction/explainability tables) —
  never the raw Silver layer, per the architecture.

### Data sources per page
| Page | Snowflake tables |
|---|---|
| 1 — SOC Overview | `MART_TRAFFIC_BY_HOUR`, `MART_ATTACK_SUMMARY`, `GOLD_SECURITY_ALERTS` |
| 2 — Attack Analysis | `MART_TRAFFIC_BY_HOUR`, `MART_ATTACK_SUMMARY`, `MART_PROTOCOL_BREAKDOWN` |
| 3 — Explainability & Monitoring | `GOLD_SHAP_GLOBAL`, `GOLD_DRIFT` |
| 4 — Model & Data Quality | `GOLD_MODEL_PERFORMANCE`, `MART_QUALITY_SUMMARY`, `GOLD_MODEL_PREDICTIONS` |

---

## 2. Page 1 — SOC Overview

**Purpose:** at-a-glance operational picture — how much traffic, how much is
malicious, and the latest critical alerts.

**Visuals:** KPI cards (Total Flows 150K, Total Alerts ~125K, Malicious Rate,
High-Severity Alerts), flows-over-time (benign vs malicious), attack-type and
DoS/Recon/Benign donuts, latest critical-alerts table.

**Talking points:**
- The **flows-over-time chart shows three distinct DDoS waves** rising above a
  calm benign baseline — the attack timeline is immediately legible.
- Ground-truth malicious rate ≈ **61%**; the model *flags* ≈ **83%** (the gap is
  the benign false-positive behavior — see Page 4).

---

## 3. Page 2 — Attack Analysis

**Purpose:** deep dive into the attacks — when, what category, which protocol,
which tool.

**Visuals:** attacks-over-time by type, DoS-vs-Reconnaissance stacked area,
attacks-by-tool bar, **Protocol × Attack-type heatmap**, attack-profile table
(rate / duration / tool per attack).

**Talking points:**
- The **recon campaign precedes the DoS waves** (visible in the DoS-vs-Recon area
  chart) — a realistic attack storyline.
- The **protocol heatmap** shows clean fingerprints: ICMPFlood is pure `icmp`,
  UDPFlood pure `udp`, scans favour `tcp`/`udp`.
- **Attack tools** identified: Hping3 (dominant), Goldeneye, Torshammer, Nmap,
  Slowloris.
- The attack-profile table separates behaviours: **floods = high rate / short
  duration**, **SlowrateDoS = low rate / long duration**.

**Caveat:** the brief's "Top destinations (hashed IPs)" visual is **not buildable**
— this version of the dataset has no IP columns (absent from the source CSV).

---

## 4. Page 3 — Explainability & Monitoring

**Purpose:** *why* the model decides, and whether incoming data is drifting.

**Visuals:** global SHAP feature importance, top-features-per-attack (with a class
slicer), drift verdict cards, PSI-by-feature bar, drift detail table.

**Talking points:**
- **Global drivers (SHAP):** `sTtl`, `Proto_udp`, `sHops`, `sMeanPktSz`,
  `State_ECO` — routing/protocol/packet-size signals, consistent with the
  attack-generation setup.
- **Drift:** **No dataset-level drift** (13/76 columns show per-column PSI > 0.2,
  below the 50% threshold). The drifted columns are the TCP-only / structural-null
  features (`SrcWin`, `DstGap`, `SrcTCPBase`), i.e. expected sensitivity, not a
  real shift.

**Caveat (good teaching point):** the KS-test p-values are ~0 for *every* feature
because with 40K vs 182K samples the test has enormous statistical power — it flags
everything as "significant." That is exactly why drift is judged on **PSI (effect
size)**, not p-value.

---

## 5. Page 4 — Model & Data Quality

**Purpose:** is the model good, and is the data trustworthy.

**Visuals:** model KPI cards (Version v1, Accuracy 76.6%, Macro-F1 92.4%),
precision & recall by class, per-class metrics table; data-quality KPIs and the
per-batch-hour validation table.

**Talking points:**
- **Macro-F1 0.924** on the held-out test set (≈182K rows, 15% of the full 1.2M
  dataset — *not* the streamed 150K, so no leakage).
- **8 of 9 classes score 0.98–1.00** precision & recall.
- The **one documented weakness is visible and quantified**: Benign recall **0.42**
  (precision 0.99) vs UDPFlood recall **0.99** (precision **0.62**). The model
  catches essentially every attack at the cost of over-flagging benign high-volume
  UDP traffic — a **false-positive problem, not a missed-attack problem**.
- **Data quality:** **100% Great Expectations pass rate** across 186 micro-batches
  / 150K rows, **0 failed expectations**.

---

## 6. Key findings (cross-cutting)

1. **The model is strong but has one honest limitation** — the Benign↔UDPFlood
   confusion — which is documented, quantified, and shown rather than hidden.
2. **The pipeline is trustworthy** — every micro-batch validated, 100% pass rate.
3. **The residual confusion is data-fundamental, not a model flaw** — it stems
   from benign and flood UDP traffic overlapping in single-flow features (see
   `modeling/notes.md`); resolving it needs cross-flow / temporal signals not in
   this dataset.

---

## 7. Honest limitations to state in the report

- **Synthetic timeline:** the source CSV has no recoverable chronological order,
  so `ingestion_timestamp` / `flow_timestamp` are **fabricated for the demo**
  (a diurnal baseline + recon/DDoS waves over 48h). Traffic-over-time and
  alerts-over-time visuals reflect this synthetic timeline, clearly labelled as
  such — not real capture time.
- **No IP/port data** → no top-talkers / geolocation visuals.
- **False-positive rate** on benign traffic is high (~59% of benign flagged),
  the deliberate trade-off for near-perfect attack recall.
- **Quality-over-time** has only ~2 hourly buckets — GX validation used real run
  time, not the synthetic spread.

---

## 8. Not yet built (possible next steps)

- **Per-alert SHAP drill-down** ("why was *this* specific flow flagged") — needs a
  per-prediction SHAP table (150K × top-k features); global + per-class SHAP is done.
- **Drift over time** — needs multiple scoring runs to trend PSI.
- **Model-performance history** — precision/recall per model *version* once a v2 exists.
