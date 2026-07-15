# 5G-NIDD Modeling Notes

Design decisions, findings, and methodological choices for the DS side of the
5G-NIDD project. Written contemporaneously to preserve reasoning for the final
report and to give collaborators visibility into what was decided and why.

---

## Contract verification (Phase 1)

The data contract in `contracts/schemas.py` was independently verified against
`Combined.csv` before being trusted for downstream work. All 9 dropped columns
were re-derived from first principles:

- **RunTime, Mean, Sum, Min, Max**: confirmed 100% identical to `Dur` via
  pairwise equality check across all 1.2M rows. Redundant.
- **sVid, dVid**: confirmed as zero-variance (nunique = 1) with 90.6% and 99.8%
  missingness respectively. No signal.
- **Unnamed: 0**: confirmed as a clean sequential integer counter starting at 0
  — a leftover pandas index, not a real feature.
- **Seq**: confirmed as bookkeeping. Max value 137,210 in a 1.2M-row file, and
  `.diff()` shows negative jumps (values decrease at points), which a real
  chronological counter would never do. Argus internal batch identifier.

All 52 raw columns accounted for. No mismatches between the contract and the
actual CSV.

---

## Null-handling policy (Phase 1)

Followed the contract's section 4 policy: `is_tcp` and `has_dst_reply`
indicators + `-1.0` sentinel fill on numeric feature columns.

Rationale for not using mean/median imputation:

- NaNs in this dataset are **structural**, not missing-at-random. `d*` columns
  (`dTos`, `dTtl`, `dHops`) are null when the destination never replied
  (~77.6% of flows). TCP-only columns (`SrcWin`, `DstWin`, `SrcTCPBase`, etc.)
  are null for non-TCP flows by definition.
- A mean-imputed `SrcWin` on a UDP flow would be a fabricated measurement.
  Sentinel + indicator lets the model distinguish "not applicable" from
  "applicable but zero" — the indicator captures the *why* of the null, the
  sentinel keeps the column numeric so models can operate on it.
- The impossible value `-1.0` was chosen because every affected feature is a
  real-world measurement bounded at ≥0. No natural row can collide with the
  sentinel.

After preprocessing, `df[numeric_feature_cols].isnull().sum().sum() == 0`.
Structural fills verified against contract predictions:
- `d*` columns all show identical 77.56% null rate (correct — same rows).
- TCP-only columns cluster at 77–85% (correct — non-TCP + no-reply combined).

---

## Split strategy (Phase 2)

Chose **70/15/15 stratified** on `Attack Type` over the more common 80/10/10.

Reasoning: with `ICMPFlood` at only 1,155 total rows (0.095% of the dataset),
an 80/10/10 split leaves ~115 test rows for the rare class — too small for
stable per-class recall metrics (individual predictions swing the number by
whole percentage points). 70/15/15 gives ~173 rows in each of val and test,
still small but workable.

`train_test_split(..., stratify=y)` used on both splits so that class
proportions are preserved across train, val, and test. Verified post-split:
`ICMPFlood` present in all three splits, distribution matches the ~40x range
seen in the raw data.

`random_state=42` throughout for reproducibility. Splits saved as parquet
under `modeling/artifacts/splits/` so every model uses identical data.

---

## Model selection rationale

Model families evaluated: Logistic Regression, Random Forest, XGBoost.
Deep learning was deliberately excluded based on three considerations:

1. Empirical evidence that tree-based models outperform deep learning on
   tabular data with moderate feature counts (Grinsztajn et al., 2022,
   NeurIPS).
2. The availability of exact and computationally efficient SHAP attributions
   for tree models, aligning with the project's explainability requirement.
3. Significantly higher engineering overhead of neural networks in handling
   severe class imbalance, which tree-based models resolve through native
   support for class weighting (`class_weight='balanced'` in sklearn,
   `sample_weight` in XGBoost).

Deep learning has legitimate applications in intrusion detection but on
different data modalities: raw packet payloads (CNNs), flow *sequences*
(recurrent / transformer models), or unsupervised autoencoders for anomaly
detection. None of these match the supervised, per-flow, feature-engineered
framing of 5G-NIDD as delivered.

---

## Metric choice — accuracy vs macro-F1

**Macro-F1 is the reported metric, not accuracy.** With `Benign` at 39% and
`UDPFlood` at 38% of the data, accuracy is dominated by the two largest
classes — a model that scores 0.99 on those and 0.00 on `ICMPFlood` would
still register ~77% accuracy. Macro-F1 averages per-class F1 equally
regardless of class size, so it reflects performance across all attack types
including rare ones. This matches the operational context: in intrusion
detection, catching a rare attack type is not a smaller success than catching
a common one.

---

## First baseline (Logistic Regression, unscaled) — diagnostic failure

Initial LogReg baseline: macro-F1 0.30, accuracy 0.51. Five of nine classes
had exactly 0.00 recall — model never predicted them at all.

Cause: feature scaling. LogReg is sensitive to feature magnitude, and the
dataset contains features spanning several orders of magnitude (`Rate` up to
500,000+, `Load` similar, byte counts in millions) alongside features bounded
in [0, 1] (`TcpRtt`, `pLoss` as fraction, `is_tcp`). Gradient descent was
dominated by the high-magnitude features, and everything else was invisible
to the optimizer. LBFGS also failed to converge.

Fix: wrapped LogReg in a `StandardScaler` pipeline. Macro-F1 jumped from
0.30 to 0.98. This was recorded as `logreg_scaled` in MLflow, with the
unscaled run preserved as a comparison. Kept as a lesson in the value of a
baseline: bad-in-a-specific-way tells you exactly what to fix.

---

## Tool-fingerprint audit (sMeanPktSz)

After scaling, LogReg feature importance showed `sMeanPktSz` with coefficient
magnitude 3× larger than the second-place feature (~16 vs ~5). Investigation
of per-class distributions revealed:

- `ICMPFlood`: mean 42.0, std 0.0 (every flow identical)
- `SYNFlood`: mean 54.0, std 0.0
- `UDPFlood`: mean 42.0, std 0.0
- `TCPConnectScan`: mean 73.9, std 1.0
- `SYNScan`: mean 58.0, std 0.4
- `Benign`: mean 105.8, std 225.6 (real variance)
- `HTTPFlood`: mean 77.4, std 60.4 (real variance)

Interpretation: attack tools (Hping3, Slowloris, etc.) are scripts that emit
uniform packet sizes, so `sMeanPktSz` is effectively a tool fingerprint for
attack classes rather than a general attack signal. Real attackers using
different tools would not produce these constant values.

Ablation: retrained LogReg without `sMeanPktSz`. Macro-F1 dropped from 0.98
to 0.97 — a 1-point loss concentrated in a single class (SYNFlood recall
1.00 → 0.87). Other classes unaffected.

Decision: **kept the feature in the pipeline**, documented as a dataset
limitation in the report. Removing a legitimately-computed feature that
carries real signal from all-but-one class would be overcorrection. The
model's reliance on this feature is minor once diagnosed; the caveat belongs
in the report, not the contract.

---

## Offset feature — file-assembly leakage (contract change)

Random Forest ranked `Offset` (documented as "Argus internal byte offset")
as its #1 feature at 11.6% importance. Per-class distribution investigation:

- `Benign`: offset range 128 → 39.7M
- `UDPFlood`: 256K → 39.3M
- `HTTPFlood`: 151K → 16.2M
- `SlowrateDoS`: 53K → 6.8M
- `SYNFlood`: 298K → 4.6M
- `SYNScan`: 4.6K → 1.1M
- `TCPConnectScan`: 6.9K → 1.1M
- `UDPScan`: 6.3K → 940K
- `ICMPFlood`: 149K → 634K

Each attack class occupies a **contiguous, non-overlapping** range of offset
values. This is not a network property — it reflects how the dataset was
constructed by concatenating per-class captures in the CSV in order. `Offset`
is effectively a file-position indicator that leaks class label with near-
perfect fidelity in this dataset, while carrying no meaningful signal in a
live streaming context.

Distinction from `sMeanPktSz`: unlike a tool fingerprint (which is at least
a real measurement of packet size), `Offset` is dataset-assembly metadata
with no plausible causal link to attack behavior. This is a *feature
representation bug*, not a caveat.

Action: `Offset` moved from `FEATURE_SCHEMA` to `DROPPED_COLUMNS` in
`contracts/schemas.py`. `SCHEMA_VERSION` bumped from 1.0.0 to 1.0.1.
DE side notified so `spark_bronze_silver.py` can stop propagating the field
if desired. All models retrained on the cleaned feature set.

Impact after removal: macro-F1 dropped from 0.98 → 0.92 across all three
model families. This is the **honest** performance ceiling for genuine
per-flow attack classification on this dataset.

---

## Baseline comparison — three-model shootout

All three models trained without `Offset`, with class weighting for the
9-class multiclass target:

| Metric        | LogReg | RandomForest | XGBoost |
|---------------|--------|--------------|---------|
| Accuracy      | 0.76   | 0.72         | 0.77    |
| Macro-F1      | 0.91   | 0.92         | 0.92    |
| Weighted-F1   | 0.74   | 0.72         | 0.75    |
| Benign recall | 0.40   | 0.49         | 0.41    |
| UDPFlood recall | 1.00 | 0.80         | 0.99    |

**Finding**: all three families converge on approximately the same macro-F1
(~0.91–0.92) despite very different function classes. This is strong evidence
that the residual confusion is **feature-fundamental**, not a limit of model
expressiveness. Different feature importance rankings across models
(RF favors numeric volume features, XGBoost favors categorical connection-
state features) show the models arrive at similar performance through
different reasoning paths.

The confusion is concentrated on a single pair: Benign ↔ UDPFlood.
Asymmetric: ~58% of Benign flows misclassified as UDPFlood; only ~0.7% of
UDPFloods misclassified as Benign (XGBoost). All other 7 attack categories
achieve 98–100% recall. In operational terms, this is a **false-positive
problem, not a missed-attack problem** — the model catches essentially every
attack, at the cost of over-flagging some legitimate high-volume UDP traffic.
This trade-off direction is favorable for intrusion detection.

---

## Post-baseline feature engineering (attempted, no improvement)

Six engineered features added specifically targeting the Benign↔UDPFlood
boundary:

- `bytes_per_packet` = TotBytes / (TotPkts + 1) — floods use uniform sizes
- `src_dst_pkt_ratio` = SrcPkts / (DstPkts + 1) — floods are near-one-way
- `src_dst_byte_ratio` = SrcBytes / (DstBytes + 1) — directionality on volume
- `log_rate` = log1p(Rate) — Rate spans 5+ orders of magnitude
- `log_totbytes` = log1p(TotBytes) — same reasoning for byte volume
- `log_totpkts` = log1p(TotPkts) — same reasoning for packet volume

Result on XGBoost: **no measurable improvement**. Macro-F1 unchanged at 0.92.
Only `src_dst_pkt_ratio` cracked the top-10 feature importance (rank 6, 5.5%).
Other engineered features were either redundant with existing signals or
provided no additional discriminative power at the Benign↔UDPFlood boundary.

Combined with the earlier three-model convergence finding, this strengthens
the case that the residual confusion is **structurally embedded in the
single-flow feature representation**. Distinguishing benign high-volume UDP
traffic from UDPFlood attacks likely requires contextual signals across
multiple flows (temporal patterns, source IP diversity, destination
reputation), which are outside the scope of a per-flow classifier.

Decision: engineered features kept in the pipeline (cost is trivial, one
feature adds minor value, documentation preserves the paper trail of what
was attempted). Reported as a negative finding — attempted, honest result.

---

## Cache invalidation convention

`modeling/artifacts/processed/` contains parquet caches of preprocessed data
to avoid re-running the ~30–60s preprocessing on every training run.
**Delete these caches whenever `contracts/schemas.py` changes** to force a
rebuild on next run. The `load_or_build()` function in `preprocessing.py`
automatically rebuilds if no cache file exists.

Two caches maintained side-by-side (baseline features vs baseline +
engineered) so that ablations can be run without editing the pipeline flag
between runs.

---

## Dimensionality reduction — considered and rejected

Principal Component Analysis, Linear Discriminant Analysis, and Locally
Linear Embedding were considered as alternative approaches to improve
Benign↔UDPFlood separability but rejected on principle. These methods
project features into lower-dimensional spaces but do not add discriminative
information not already present in the original representation. Since
three model families (linear, bagged trees, gradient-boosted trees) already
achieved similar performance on the raw features, and targeted engineered
features (ratios, log-transforms) produced no improvement, the residual
confusion is attributable to genuine overlap of the two classes in the
single-flow feature space rather than to representation choice. No
supervised linear projection (LDA) can separate classes that overlap in
the original feature space, and PCA preserves variance without regard to
class boundaries.

The information required to resolve the confusion (temporal aggregation,
source diversity, cross-flow correlation) is not derivable from single-flow
summary statistics and requires an architecturally different modeling
framing beyond the scope of this project.

## Binary vs multiclass framing

A binary XGBoost model (Benign vs Malicious) was trained to test whether
reframing the task as a binary decision would resolve the Benign↔UDPFlood
confusion in the multiclass model. Result: the binary model achieved
ROC-AUC 0.87, materially worse than the multiclass model's implicit binary
performance (0.996 malicious recall, 0.42 benign recall — corresponding
to a much stronger separation).

Diagnostic: the binary model concentrated 48% of feature importance on
a single TCP-only feature (SrcGap), suggesting it exploited a shortcut
separating TCP-heavy attack traffic from mixed benign traffic. The
multiclass model was forced to learn distinct patterns for each attack
type (dominated by State_ECO, Proto_udp, State_INT connection-state
signals) and its aggregated behavior is a stronger implicit binary
classifier.

This is a case where multiclass framing improves binary performance:
the constraint of distinguishing between attack types forces the model
to learn richer discriminative features than a directly-trained binary
classifier will discover on its own. As a consequence, both the standalone
binary framing and the hierarchical two-stage approach (binary → then
multiclass) were rejected in favor of the direct multiclass classifier.

## Specialist ablation — confirmed ceiling

The specialist Benign↔UDPFlood classifier was retrained without sMeanPktSz
to test whether removing the dominant feature would force it to learn a
richer pattern (analogous to how multiclass training forces feature
diversity).

Result: metrics were identical to two decimal places (ROC-AUC 0.7942 in
both runs; Benign recall 0.42; UDPFlood recall 0.98; F1 0.76). Only the
feature-importance distribution changed: bytes_per_packet (TotBytes/TotPkts,
an engineered feature) took 82% of the importance, replacing sMeanPktSz.
Both features are packet-size proxies derivable from the same underlying
quantities.

This confirms the ceiling result more strongly than any prior experiment:
the Benign↔UDPFlood confusion is not attributable to any specific feature.
The tool-fingerprint / traffic-similarity signal is embedded structurally
in per-flow measurements, and any feature capturing packet-size information
carries it. Removing one such feature causes the model to fall back on
another, arriving at the same decision boundary. No amount of feature
selection can lift the boundary because the discriminative information
required is not present in single-flow data — it exists only in
cross-flow contextual signals not measured in this dataset.

## Model space exploration

Beyond the three families reported (Logistic Regression, Random Forest,
XGBoost), other supervised classification approaches (LightGBM, CatBoost,
Support Vector Machines, k-Nearest Neighbors, deep tabular neural networks)
were considered but not evaluated. This decision was based on the consistent
performance ceiling observed across the three primary models on the specific
subtask (Benign↔UDPFlood specialist ROC-AUC 0.79, identical whether or not
the dominant packet-size feature was included). Since the ceiling stems from
overlapping class distributions in per-flow feature space rather than any
one model's expressiveness limits, alternative supervised classifiers would
be expected to plateau at the same performance level. Improving results
beyond this ceiling requires an architectural change in problem framing —
specifically incorporating temporal or cross-flow contextual signals not
available in the single-flow dataset used here — rather than substituting
one supervised classifier for another.

## Manual class weights — final confirmation of the trade-off curve

A final attempt to improve Benign recall without sacrificing UDPFlood used
manual class weights instead of sample_weight='balanced' (Benign: 1.5,
UDPFlood: 0.7, rare attack classes: 3-10). Result: Benign recall reached
1.00 while UDPFlood recall collapsed to 0.23 — the model now misses 77% of
UDP floods.

This confirms with the highest possible confidence that the Benign↔UDPFlood
trade-off is not addressable by any means available within the current data.
Across five mechanisms — model architecture, feature engineering,
inference-time thresholding, binary reframing, and training-time class
weighting — every configuration produces a different operating point on
the same trade-off curve. The curve does not shift; only the operating
point moves along it. Any Benign recall improvement comes at exact
proportional cost to UDPFlood recall.

Decision: production model uses XGBoost multiclass with sample_weight=
'balanced'. This operating point (Benign 0.41, UDPFlood 0.996) is chosen
because in intrusion detection, near-perfect attack recall is the primary
operational requirement, and Benign false positives can be handled by
downstream analyst review or contextual filtering.

## References

- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based
  models still outperform deep learning on typical tabular data?* NeurIPS.
  Referenced for the deep-learning-exclusion rationale.