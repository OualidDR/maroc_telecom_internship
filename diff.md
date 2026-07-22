git diff HEAD~1 HEAD
diff --git a/.gitignore b/.gitignore
index 8291fdb..d2ab6ce 100644
--- a/.gitignore
+++ b/.gitignore
@@ -21,4 +21,21 @@ metastore_db/
 # OS / editor cruft
 .DS_Store
 .vscode/
-.idea/
\ No newline at end of file
+.idea/
+
+# Jupyter
+.ipynb_checkpoints/
+modeling/notebooks/
+
+modeling/artifacts/
+
+
+## Ignore the following files
+confusion_matrix.png
+feature_importance.png
+threshold_analysis.png
+
+# MLflow local state
+mlruns/
+mlflow.db
+mlartifacts/
diff --git a/contracts/schemas.py b/contracts/schemas.py
index 3df31e8..06fec5a 100644
--- a/contracts/schemas.py
+++ b/contracts/schemas.py
@@ -36,6 +36,16 @@ DROPPED_COLUMNS = {
     "Max": "100% identical to Dur on every row -- redundant.",
     "sVid": "Zero variance (constant = 610 where not null) and >90% missing -- no signal.",
     "dVid": "Zero variance (constant = 610 where not null) and >99% missing -- no signal.",
+    "Offset": (
+    "Argus internal byte offset into the source capture. In Combined.csv, "
+    "attack classes occupy contiguous non-overlapping offset ranges "
+    "(e.g. ICMPFlood: 149K-634K only; UDPScan: 6K-940K only) because the "
+    "dataset was assembled by concatenating per-class captures in order. "
+    "This makes Offset a near-perfect leaky proxy for Attack Type in this "
+    "dataset, while carrying no meaningful signal in production streaming. "
+    "Detected via RF feature importance ranking Offset #1 (11.6%) despite "
+    "having no plausible causal link to attack behavior."
+),
 }
 
 
@@ -100,7 +110,6 @@ FEATURE_SCHEMA: list[FeatureSpec] = [
     FeatureSpec("TotBytes", "int", False, "Total bytes, both directions."),
     FeatureSpec("SrcBytes", "int", False, "Bytes sent by source."),
     FeatureSpec("DstBytes", "int", False, "Bytes sent by destination."),
-    FeatureSpec("Offset", "int", False, "Argus internal byte offset."),
     FeatureSpec("sMeanPktSz", "float", False, "Mean packet size, source side."),
     FeatureSpec("dMeanPktSz", "float", False, "Mean packet size, destination side."),
 
@@ -164,7 +173,7 @@ EVENT_WRAPPER_EXAMPLE = {
     "flow": "{...39 raw feature fields + 3 label fields...}",
 }
 
-SCHEMA_VERSION = "1.0.0"
+SCHEMA_VERSION = "1.0.1"
 
 
 if __name__ == "__main__":
diff --git a/modeling/notes.md b/modeling/notes.md
new file mode 100644
index 0000000..b8d0f58
--- /dev/null
+++ b/modeling/notes.md
@@ -0,0 +1,412 @@
+# 5G-NIDD Modeling Notes
+
+Design decisions, findings, and methodological choices for the DS side of the
+5G-NIDD project. Written contemporaneously to preserve reasoning for the final
+report and to give collaborators visibility into what was decided and why.
+
+---
+
+## Contract verification (Phase 1)
+
+The data contract in `contracts/schemas.py` was independently verified against
+`Combined.csv` before being trusted for downstream work. All 9 dropped columns
+were re-derived from first principles:
+
+- **RunTime, Mean, Sum, Min, Max**: confirmed 100% identical to `Dur` via
+  pairwise equality check across all 1.2M rows. Redundant.
+- **sVid, dVid**: confirmed as zero-variance (nunique = 1) with 90.6% and 99.8%
+  missingness respectively. No signal.
+- **Unnamed: 0**: confirmed as a clean sequential integer counter starting at 0
+  — a leftover pandas index, not a real feature.
+- **Seq**: confirmed as bookkeeping. Max value 137,210 in a 1.2M-row file, and
+  `.diff()` shows negative jumps (values decrease at points), which a real
+  chronological counter would never do. Argus internal batch identifier.
+
+All 52 raw columns accounted for. No mismatches between the contract and the
+actual CSV.
+
+---
+
+## Null-handling policy (Phase 1)
+
+Followed the contract's section 4 policy: `is_tcp` and `has_dst_reply`
+indicators + `-1.0` sentinel fill on numeric feature columns.
+
+Rationale for not using mean/median imputation:
+
+- NaNs in this dataset are **structural**, not missing-at-random. `d*` columns
+  (`dTos`, `dTtl`, `dHops`) are null when the destination never replied
+  (~77.6% of flows). TCP-only columns (`SrcWin`, `DstWin`, `SrcTCPBase`, etc.)
+  are null for non-TCP flows by definition.
+- A mean-imputed `SrcWin` on a UDP flow would be a fabricated measurement.
+  Sentinel + indicator lets the model distinguish "not applicable" from
+  "applicable but zero" — the indicator captures the *why* of the null, the
+  sentinel keeps the column numeric so models can operate on it.
+- The impossible value `-1.0` was chosen because every affected feature is a
+  real-world measurement bounded at ≥0. No natural row can collide with the
+  sentinel.
+
+After preprocessing, `df[numeric_feature_cols].isnull().sum().sum() == 0`.
+Structural fills verified against contract predictions:
+- `d*` columns all show identical 77.56% null rate (correct — same rows).
+- TCP-only columns cluster at 77–85% (correct — non-TCP + no-reply combined).
+
+---
+
+## Split strategy (Phase 2)
+
+Chose **70/15/15 stratified** on `Attack Type` over the more common 80/10/10.
+
+Reasoning: with `ICMPFlood` at only 1,155 total rows (0.095% of the dataset),
+an 80/10/10 split leaves ~115 test rows for the rare class — too small for
+stable per-class recall metrics (individual predictions swing the number by
+whole percentage points). 70/15/15 gives ~173 rows in each of val and test,
+still small but workable.
+
+`train_test_split(..., stratify=y)` used on both splits so that class
+proportions are preserved across train, val, and test. Verified post-split:
+`ICMPFlood` present in all three splits, distribution matches the ~40x range
+seen in the raw data.
+
+`random_state=42` throughout for reproducibility. Splits saved as parquet
+under `modeling/artifacts/splits/` so every model uses identical data.
+
+---
+
+## Model selection rationale
+
+Model families evaluated: Logistic Regression, Random Forest, XGBoost.
+Deep learning was deliberately excluded based on three considerations:
+
+1. Empirical evidence that tree-based models outperform deep learning on
+   tabular data with moderate feature counts (Grinsztajn et al., 2022,
+   NeurIPS).
+2. The availability of exact and computationally efficient SHAP attributions
+   for tree models, aligning with the project's explainability requirement.
+3. Significantly higher engineering overhead of neural networks in handling
+   severe class imbalance, which tree-based models resolve through native
+   support for class weighting (`class_weight='balanced'` in sklearn,
+   `sample_weight` in XGBoost).
+
+Deep learning has legitimate applications in intrusion detection but on
+different data modalities: raw packet payloads (CNNs), flow *sequences*
+(recurrent / transformer models), or unsupervised autoencoders for anomaly
+detection. None of these match the supervised, per-flow, feature-engineered
+framing of 5G-NIDD as delivered.
+
+---
+
+## Metric choice — accuracy vs macro-F1
+
+**Macro-F1 is the reported metric, not accuracy.** With `Benign` at 39% and
+`UDPFlood` at 38% of the data, accuracy is dominated by the two largest
+classes — a model that scores 0.99 on those and 0.00 on `ICMPFlood` would
+still register ~77% accuracy. Macro-F1 averages per-class F1 equally
+regardless of class size, so it reflects performance across all attack types
+including rare ones. This matches the operational context: in intrusion
+detection, catching a rare attack type is not a smaller success than catching
+a common one.
+
+---
+
+## First baseline (Logistic Regression, unscaled) — diagnostic failure
+
+Initial LogReg baseline: macro-F1 0.30, accuracy 0.51. Five of nine classes
+had exactly 0.00 recall — model never predicted them at all.
+
+Cause: feature scaling. LogReg is sensitive to feature magnitude, and the
+dataset contains features spanning several orders of magnitude (`Rate` up to
+500,000+, `Load` similar, byte counts in millions) alongside features bounded
+in [0, 1] (`TcpRtt`, `pLoss` as fraction, `is_tcp`). Gradient descent was
+dominated by the high-magnitude features, and everything else was invisible
+to the optimizer. LBFGS also failed to converge.
+
+Fix: wrapped LogReg in a `StandardScaler` pipeline. Macro-F1 jumped from
+0.30 to 0.98. This was recorded as `logreg_scaled` in MLflow, with the
+unscaled run preserved as a comparison. Kept as a lesson in the value of a
+baseline: bad-in-a-specific-way tells you exactly what to fix.
+
+---
+
+## Tool-fingerprint audit (sMeanPktSz)
+
+After scaling, LogReg feature importance showed `sMeanPktSz` with coefficient
+magnitude 3× larger than the second-place feature (~16 vs ~5). Investigation
+of per-class distributions revealed:
+
+- `ICMPFlood`: mean 42.0, std 0.0 (every flow identical)
+- `SYNFlood`: mean 54.0, std 0.0
+- `UDPFlood`: mean 42.0, std 0.0
+- `TCPConnectScan`: mean 73.9, std 1.0
+- `SYNScan`: mean 58.0, std 0.4
+- `Benign`: mean 105.8, std 225.6 (real variance)
+- `HTTPFlood`: mean 77.4, std 60.4 (real variance)
+
+Interpretation: attack tools (Hping3, Slowloris, etc.) are scripts that emit
+uniform packet sizes, so `sMeanPktSz` is effectively a tool fingerprint for
+attack classes rather than a general attack signal. Real attackers using
+different tools would not produce these constant values.
+
+Ablation: retrained LogReg without `sMeanPktSz`. Macro-F1 dropped from 0.98
+to 0.97 — a 1-point loss concentrated in a single class (SYNFlood recall
+1.00 → 0.87). Other classes unaffected.
+
+Decision: **kept the feature in the pipeline**, documented as a dataset
+limitation in the report. Removing a legitimately-computed feature that
+carries real signal from all-but-one class would be overcorrection. The
+model's reliance on this feature is minor once diagnosed; the caveat belongs
+in the report, not the contract.
+
+---
+
+## Offset feature — file-assembly leakage (contract change)
+
+Random Forest ranked `Offset` (documented as "Argus internal byte offset")
+as its #1 feature at 11.6% importance. Per-class distribution investigation:
+
+- `Benign`: offset range 128 → 39.7M
+- `UDPFlood`: 256K → 39.3M
+- `HTTPFlood`: 151K → 16.2M
+- `SlowrateDoS`: 53K → 6.8M
+- `SYNFlood`: 298K → 4.6M
+- `SYNScan`: 4.6K → 1.1M
+- `TCPConnectScan`: 6.9K → 1.1M
+- `UDPScan`: 6.3K → 940K
+- `ICMPFlood`: 149K → 634K
+
+Each attack class occupies a **contiguous, non-overlapping** range of offset
+values. This is not a network property — it reflects how the dataset was
+constructed by concatenating per-class captures in the CSV in order. `Offset`
+is effectively a file-position indicator that leaks class label with near-
+perfect fidelity in this dataset, while carrying no meaningful signal in a
+live streaming context.
+
+Distinction from `sMeanPktSz`: unlike a tool fingerprint (which is at least
+a real measurement of packet size), `Offset` is dataset-assembly metadata
+with no plausible causal link to attack behavior. This is a *feature
+representation bug*, not a caveat.
+
+Action: `Offset` moved from `FEATURE_SCHEMA` to `DROPPED_COLUMNS` in
+`contracts/schemas.py`. `SCHEMA_VERSION` bumped from 1.0.0 to 1.0.1.
+DE side notified so `spark_bronze_silver.py` can stop propagating the field
+if desired. All models retrained on the cleaned feature set.
+
+Impact after removal: macro-F1 dropped from 0.98 → 0.92 across all three
+model families. This is the **honest** performance ceiling for genuine
+per-flow attack classification on this dataset.
+
+---
+
+## Baseline comparison — three-model shootout
+
+All three models trained without `Offset`, with class weighting for the
+9-class multiclass target:
+
+| Metric        | LogReg | RandomForest | XGBoost |
+|---------------|--------|--------------|---------|
+| Accuracy      | 0.76   | 0.72         | 0.77    |
+| Macro-F1      | 0.91   | 0.92         | 0.92    |
+| Weighted-F1   | 0.74   | 0.72         | 0.75    |
+| Benign recall | 0.40   | 0.49         | 0.41    |
+| UDPFlood recall | 1.00 | 0.80         | 0.99    |
+
+**Finding**: all three families converge on approximately the same macro-F1
+(~0.91–0.92) despite very different function classes. This is strong evidence
+that the residual confusion is **feature-fundamental**, not a limit of model
+expressiveness. Different feature importance rankings across models
+(RF favors numeric volume features, XGBoost favors categorical connection-
+state features) show the models arrive at similar performance through
+different reasoning paths.
+
+The confusion is concentrated on a single pair: Benign ↔ UDPFlood.
+Asymmetric: ~58% of Benign flows misclassified as UDPFlood; only ~0.7% of
+UDPFloods misclassified as Benign (XGBoost). All other 7 attack categories
+achieve 98–100% recall. In operational terms, this is a **false-positive
+problem, not a missed-attack problem** — the model catches essentially every
+attack, at the cost of over-flagging some legitimate high-volume UDP traffic.
+This trade-off direction is favorable for intrusion detection.
+
+---
+
+## Post-baseline feature engineering (attempted, no improvement)
+
+Six engineered features added specifically targeting the Benign↔UDPFlood
+boundary:
+
+- `bytes_per_packet` = TotBytes / (TotPkts + 1) — floods use uniform sizes
+- `src_dst_pkt_ratio` = SrcPkts / (DstPkts + 1) — floods are near-one-way
+- `src_dst_byte_ratio` = SrcBytes / (DstBytes + 1) — directionality on volume
+- `log_rate` = log1p(Rate) — Rate spans 5+ orders of magnitude
+- `log_totbytes` = log1p(TotBytes) — same reasoning for byte volume
+- `log_totpkts` = log1p(TotPkts) — same reasoning for packet volume
+
+Result on XGBoost: **no measurable improvement**. Macro-F1 unchanged at 0.92.
+Only `src_dst_pkt_ratio` cracked the top-10 feature importance (rank 6, 5.5%).
+Other engineered features were either redundant with existing signals or
+provided no additional discriminative power at the Benign↔UDPFlood boundary.
+
+Combined with the earlier three-model convergence finding, this strengthens
+the case that the residual confusion is **structurally embedded in the
+single-flow feature representation**. Distinguishing benign high-volume UDP
+traffic from UDPFlood attacks likely requires contextual signals across
+multiple flows (temporal patterns, source IP diversity, destination
+reputation), which are outside the scope of a per-flow classifier.
+
+Decision: engineered features kept in the pipeline (cost is trivial, one
+feature adds minor value, documentation preserves the paper trail of what
+was attempted). Reported as a negative finding — attempted, honest result.
+
+---
+
+## Cache invalidation convention
+
+`modeling/artifacts/processed/` contains parquet caches of preprocessed data
+to avoid re-running the ~30–60s preprocessing on every training run.
+**Delete these caches whenever `contracts/schemas.py` changes** to force a
+rebuild on next run. The `load_or_build()` function in `preprocessing.py`
+automatically rebuilds if no cache file exists.
+
+Two caches maintained side-by-side (baseline features vs baseline +
+engineered) so that ablations can be run without editing the pipeline flag
+between runs.
+
+---
+
+## Dimensionality reduction — considered and rejected
+
+Principal Component Analysis, Linear Discriminant Analysis, and Locally
+Linear Embedding were considered as alternative approaches to improve
+Benign↔UDPFlood separability but rejected on principle. These methods
+project features into lower-dimensional spaces but do not add discriminative
+information not already present in the original representation. Since
+three model families (linear, bagged trees, gradient-boosted trees) already
+achieved similar performance on the raw features, and targeted engineered
+features (ratios, log-transforms) produced no improvement, the residual
+confusion is attributable to genuine overlap of the two classes in the
+single-flow feature space rather than to representation choice. No
+supervised linear projection (LDA) can separate classes that overlap in
+the original feature space, and PCA preserves variance without regard to
+class boundaries.
+
+The information required to resolve the confusion (temporal aggregation,
+source diversity, cross-flow correlation) is not derivable from single-flow
+summary statistics and requires an architecturally different modeling
+framing beyond the scope of this project.
+
+## Binary vs multiclass framing
+
+A binary XGBoost model (Benign vs Malicious) was trained to test whether
+reframing the task as a binary decision would resolve the Benign↔UDPFlood
+confusion in the multiclass model. Result: the binary model achieved
+ROC-AUC 0.87, materially worse than the multiclass model's implicit binary
+performance (0.996 malicious recall, 0.42 benign recall — corresponding
+to a much stronger separation).
+
+Diagnostic: the binary model concentrated 48% of feature importance on
+a single TCP-only feature (SrcGap), suggesting it exploited a shortcut
+separating TCP-heavy attack traffic from mixed benign traffic. The
+multiclass model was forced to learn distinct patterns for each attack
+type (dominated by State_ECO, Proto_udp, State_INT connection-state
+signals) and its aggregated behavior is a stronger implicit binary
+classifier.
+
+This is a case where multiclass framing improves binary performance:
+the constraint of distinguishing between attack types forces the model
+to learn richer discriminative features than a directly-trained binary
+classifier will discover on its own. As a consequence, both the standalone
+binary framing and the hierarchical two-stage approach (binary → then
+multiclass) were rejected in favor of the direct multiclass classifier.
+
+## Specialist ablation — confirmed ceiling
+
+The specialist Benign↔UDPFlood classifier was retrained without sMeanPktSz
+to test whether removing the dominant feature would force it to learn a
+richer pattern (analogous to how multiclass training forces feature
+diversity).
+
+Result: metrics were identical to two decimal places (ROC-AUC 0.7942 in
+both runs; Benign recall 0.42; UDPFlood recall 0.98; F1 0.76). Only the
+feature-importance distribution changed: bytes_per_packet (TotBytes/TotPkts,
+an engineered feature) took 82% of the importance, replacing sMeanPktSz.
+Both features are packet-size proxies derivable from the same underlying
+quantities.
+
+This confirms the ceiling result more strongly than any prior experiment:
+the Benign↔UDPFlood confusion is not attributable to any specific feature.
+The tool-fingerprint / traffic-similarity signal is embedded structurally
+in per-flow measurements, and any feature capturing packet-size information
+carries it. Removing one such feature causes the model to fall back on
+another, arriving at the same decision boundary. No amount of feature
+selection can lift the boundary because the discriminative information
+required is not present in single-flow data — it exists only in
+cross-flow contextual signals not measured in this dataset.
+
+## Model space exploration
+
+Beyond the three families reported (Logistic Regression, Random Forest,
+XGBoost), other supervised classification approaches (LightGBM, CatBoost,
+Support Vector Machines, k-Nearest Neighbors, deep tabular neural networks)
+were considered but not evaluated. This decision was based on the consistent
+performance ceiling observed across the three primary models on the specific
+subtask (Benign↔UDPFlood specialist ROC-AUC 0.79, identical whether or not
+the dominant packet-size feature was included). Since the ceiling stems from
+overlapping class distributions in per-flow feature space rather than any
+one model's expressiveness limits, alternative supervised classifiers would
+be expected to plateau at the same performance level. Improving results
+beyond this ceiling requires an architectural change in problem framing —
+specifically incorporating temporal or cross-flow contextual signals not
+available in the single-flow dataset used here — rather than substituting
+one supervised classifier for another.
+
+## Manual class weights — final confirmation of the trade-off curve
+
+A final attempt to improve Benign recall without sacrificing UDPFlood used
+manual class weights instead of sample_weight='balanced' (Benign: 1.5,
+UDPFlood: 0.7, rare attack classes: 3-10). Result: Benign recall reached
+1.00 while UDPFlood recall collapsed to 0.23 — the model now misses 77% of
+UDP floods.
+
+This confirms with the highest possible confidence that the Benign↔UDPFlood
+trade-off is not addressable by any means available within the current data.
+Across five mechanisms — model architecture, feature engineering,
+inference-time thresholding, binary reframing, and training-time class
+weighting — every configuration produces a different operating point on
+the same trade-off curve. The curve does not shift; only the operating
+point moves along it. Any Benign recall improvement comes at exact
+proportional cost to UDPFlood recall.
+
+Decision: production model uses XGBoost multiclass with sample_weight=
+'balanced'. This operating point (Benign 0.41, UDPFlood 0.996) is chosen
+because in intrusion detection, near-perfect attack recall is the primary
+operational requirement, and Benign false positives can be handled by
+downstream analyst review or contextual filtering.
+
+## sHops distribution — confirms Benign↔UDPFlood mechanism
+
+Post-hoc SHAP investigation of sHops (source-hop count) revealed:
+
+- All flood/DoS attack classes concentrated at sHops = 1 (zero std):
+  HTTPFlood, ICMPFlood, SlowrateDoS, UDPFlood
+- Scan attack classes concentrated at sHops ~17 (std ~6-7): SYNScan, UDPScan
+- Benign traffic spans the full range (mean 3.0, std 3.1, range 0-28),
+  overlapping with both attack regions
+
+The subset of Benign traffic with sHops = 1 is behaviorally indistinguishable
+from flood attacks on this feature. Combined with the earlier finding that
+Proto_udp is the second SHAP driver of Benign→UDPFlood misclassification,
+this explains the confusion mechanically: benign UDP flows one hop from the
+observer occupy the same region of (Proto_udp, sHops) space as UDPFloods,
+with no discriminating signal in single-flow measurements to separate them.
+
+This finding also raises a dataset-representativeness caveat: the lab
+generation placed flooders at 1 hop and scanners at ~17 hops. Real-world
+attackers using distributed botnets could produce floods from arbitrary
+hop counts, which the model has learned to associate with benign traffic.
+This is not addressable within the current dataset but should be flagged as
+a limitation of production deployment.
+
+## References
+
+- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based
+  models still outperform deep learning on typical tabular data?* NeurIPS.
+  Referenced for the deep-learning-exclusion rationale.
\ No newline at end of file
diff --git a/modeling/src/__init__.py b/modeling/src/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/modeling/src/analysis.py b/modeling/src/analysis.py
new file mode 100644
index 0000000..f38d1fd
--- /dev/null
+++ b/modeling/src/analysis.py
@@ -0,0 +1,203 @@
+"""
+Explainability analysis for the registered XGBoost model.
+
+Uses SHAP (SHapley Additive exPlanations) with the exact TreeExplainer for
+per-class global summaries and per-flow local explanations. These outputs
+serve two purposes:
+
+1. Report: per-class feature-importance plots that go beyond the aggregate
+   feature_importances_ vector, showing which features drive each attack
+   type distinctly.
+2. Serving: per-flow explanations returned alongside predictions by the
+   FastAPI layer (Phase 6), so an analyst reviewing a flagged flow can see
+   why the model made the call.
+"""
+
+import os
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
+
+import mlflow
+import mlflow.xgboost
+import numpy as np
+import pandas as pd
+import shap
+import matplotlib.pyplot as plt
+from sklearn.preprocessing import LabelEncoder
+
+
+MODEL_URI = "models:/5g-nidd-attack-classifier@staging"
+SPLITS_DIR = Path("modeling/artifacts/splits")
+OUT_DIR = Path("modeling/artifacts/shap")
+
+
+def load_model_and_data():
+    """Load the registered model and the val set from parquet."""
+    mlflow.set_tracking_uri("sqlite:///mlflow.db")
+
+    model = mlflow.xgboost.load_model(MODEL_URI)
+    X_val = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
+    y_val = pd.read_parquet(SPLITS_DIR / "y_val.parquet").squeeze()
+    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()
+
+    # Refit encoder so we can map class indices back to names
+    le = LabelEncoder().fit(y_train)
+    return model, X_val, y_val, le
+
+
+def compute_shap_values(model, X_sample):
+    """Compute SHAP values using XGBoost's native pred_contribs.
+
+    Bypasses shap.TreeExplainer's XGBoost parser, which fails on recent
+    XGBoost multiclass models (per-class base_score returned as JSON array).
+    XGBoost's built-in SHAP support always stays in sync with its own format.
+    """
+    import xgboost as xgb
+
+    booster = model.get_booster()
+    dmatrix = xgb.DMatrix(X_sample)
+
+    # pred_contribs returns:
+    # - Multiclass: shape (n_samples, n_classes, n_features + 1)
+    # - Binary: shape (n_samples, n_features + 1)
+    # The last feature column is the bias (base_score) contribution.
+    contribs = booster.predict(dmatrix, pred_contribs=True)
+
+    # Strip the bias column and reshape to the shap-library layout:
+    # (n_samples, n_features, n_classes)
+    shap_values = contribs[:, :, :-1].transpose(0, 2, 1)
+
+    return shap_values
+
+def plot_global_summary(shap_values, X_sample, le, out_dir: Path):
+    """Save a per-class SHAP summary bar plot.
+
+    Aggregates absolute SHAP values across the sample to show which features
+    the model relies on most for each class.
+    """
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    # shap.summary_plot with plot_type='bar' handles multiclass natively
+    fig = plt.figure()
+    shap.summary_plot(
+        shap_values,
+        X_sample,
+        class_names=list(le.classes_),
+        plot_type="bar",
+        show=False,
+        max_display=15,
+    )
+    plt.tight_layout()
+    plt.savefig(out_dir / "shap_global_bar.png", dpi=100, bbox_inches="tight")
+    plt.close()
+    print(f"Saved {out_dir / 'shap_global_bar.png'}")
+
+
+def plot_per_class_beeswarm(shap_values, X_sample, le, out_dir: Path):
+    """Save a beeswarm per class — shows feature effect direction and magnitude.
+
+    A beeswarm reveals not just 'this feature is important' but 'high values
+    of this feature push predictions toward this class' — direction + effect
+    size, which the bar plot doesn't show.
+    """
+    out_dir.mkdir(parents=True, exist_ok=True)
+
+    for i, class_name in enumerate(le.classes_):
+        # shap_values here is a 3D array indexed [:, :, class_idx]
+        class_shap = shap_values[:, :, i]
+
+        fig = plt.figure()
+        shap.summary_plot(
+            class_shap,
+            X_sample,
+            show=False,
+            max_display=15,
+        )
+        plt.title(f"SHAP feature effects for class: {class_name}")
+        plt.tight_layout()
+        plt.savefig(
+            out_dir / f"shap_beeswarm_{class_name}.png",
+            dpi=100,
+            bbox_inches="tight",
+        )
+        plt.close()
+
+    print(f"Saved {len(le.classes_)} per-class beeswarm plots to {out_dir}")
+
+
+def local_explanation(model, X_row: pd.DataFrame, le, top_k: int = 5):
+    """Explain a single prediction using XGBoost's native pred_contribs."""
+    import xgboost as xgb
+
+    booster = model.get_booster()
+    dmatrix = xgb.DMatrix(X_row)
+    contribs = booster.predict(dmatrix, pred_contribs=True)
+
+    # Shape: (1, n_classes, n_features + 1) — strip bias
+    contribs = contribs[:, :, :-1]
+
+    proba = model.predict_proba(X_row)[0]
+    pred_idx = int(np.argmax(proba))
+    pred_class = le.classes_[pred_idx]
+
+    # SHAP values for the predicted class only
+    class_shap = contribs[0, pred_idx, :]
+
+    feature_names = X_row.columns
+    top_k_idx = np.argsort(np.abs(class_shap))[-top_k:][::-1]
+
+    top_features = [
+        {
+            "feature": feature_names[i],
+            "value": float(X_row.iloc[0, i]),
+            "shap_contribution": float(class_shap[i]),
+        }
+        for i in top_k_idx
+    ]
+
+    return {
+        "predicted_class": pred_class,
+        "probability": float(proba[pred_idx]),
+        "top_features": top_features,
+    }
+
+def main():
+    print("Loading model and data...")
+    model, X_val, y_val, le = load_model_and_data()
+
+    # SHAP on full val (182K rows) would take an hour+. Sample for global plots.
+    print("Sampling 2000 rows for global SHAP...")
+    sample_idx = X_val.sample(2000, random_state=42).index
+    X_sample = X_val.loc[sample_idx]
+    y_sample = y_val.loc[sample_idx]
+
+    print("Computing SHAP values (may take 1-2 min)...")
+    shap_values = compute_shap_values(model, X_sample)
+    print(f"SHAP values shape: {shap_values.shape}")
+
+    print("\nGenerating global summary bar plot...")
+    plot_global_summary(shap_values, X_sample, le, OUT_DIR)
+
+    print("Generating per-class beeswarm plots...")
+    plot_per_class_beeswarm(shap_values, X_sample, le, OUT_DIR)
+
+    print("\nDemo local explanation on 3 random val flows...")
+    for i, idx in enumerate(X_val.sample(3, random_state=1).index):
+        X_row = X_val.loc[[idx]]
+        true_label = y_val.loc[idx]
+        explanation = local_explanation(model, X_row, le, top_k=5)
+
+        print(f"\n--- Flow #{i+1} ---")
+        print(f"  True label:      {true_label}")
+        print(f"  Predicted:       {explanation['predicted_class']} "
+              f"(prob {explanation['probability']:.3f})")
+        print(f"  Top features pushing toward predicted class:")
+        for f in explanation['top_features']:
+            direction = "↑" if f['shap_contribution'] > 0 else "↓"
+            print(f"    {direction} {f['feature']:25s} "
+                  f"value={f['value']:>12.4f}  shap={f['shap_contribution']:+.4f}")
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
diff --git a/modeling/src/eda.py b/modeling/src/eda.py
new file mode 100644
index 0000000..48ccfd0
--- /dev/null
+++ b/modeling/src/eda.py
@@ -0,0 +1,10 @@
+import sys
+from pathlib import Path
+sys.path.append(str(Path(__file__).resolve().parent.parent))
+
+from src.preprocessing import DEFAULT_CSV_PATH, load_or_build
+
+df = load_or_build(DEFAULT_CSV_PATH)
+print(f"Shape after preprocessing: {df.shape}")
+print(df["Attack Type"].value_counts())
+
diff --git a/modeling/src/feature_engineering.py b/modeling/src/feature_engineering.py
new file mode 100644
index 0000000..e038369
--- /dev/null
+++ b/modeling/src/feature_engineering.py
@@ -0,0 +1,52 @@
+"""
+Engineered features targeting the Benign vs UDPFlood confusion.
+
+Rationale:
+The three baseline models (LogReg, RF, XGBoost) all converge on the same
+trade-off at this class boundary, suggesting the confusion is fundamental
+to the feature representation rather than a model expressiveness limit.
+These engineered features are designed to capture behavioral signals that
+distinguish flood traffic (uniform, one-directional, high-rate) from heavy
+legitimate UDP traffic (variable, bidirectional).
+
+Added features:
+- bytes_per_packet:      Total bytes / total packets. Floods often use
+                         uniform small packets; legitimate traffic varies.
+- src_dst_pkt_ratio:     SrcPkts / DstPkts. Floods are near-one-way (attacker
+                         pumps, destination silent); legitimate flows balance.
+- src_dst_byte_ratio:    SrcBytes / DstBytes. Same directionality signal on
+                         a volume basis.
+- log_rate:              log1p(Rate). Rate spans 5+ orders of magnitude;
+                         log-scale exposes low-value patterns to the model.
+- log_totbytes:          log1p(TotBytes). Same reasoning for byte volume.
+- log_totpkts:           log1p(TotPkts). Same reasoning for packet volume.
+
+All ratios use +1 denominator smoothing to avoid division by zero on flows
+where one side sent nothing.
+"""
+
+import numpy as np
+import pandas as pd
+
+
+def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
+    """Add engineered features that target the Benign vs UDPFlood boundary.
+
+    Called after load_and_clean() so all sentinel fills and indicators are
+    already in place. Sentinel values (-1) will propagate through ratios;
+    that's acceptable because the is_tcp / has_dst_reply indicators still
+    tell the model which rows those sentinels came from.
+    """
+    df = df.copy()
+
+    # Ratios (directionality and packet-size uniformity)
+    df["bytes_per_packet"] = df["TotBytes"] / (df["TotPkts"] + 1)
+    df["src_dst_pkt_ratio"] = df["SrcPkts"] / (df["DstPkts"] + 1)
+    df["src_dst_byte_ratio"] = df["SrcBytes"] / (df["DstBytes"] + 1)
+
+    # Log-scaled volume features (compress skewed distributions)
+    df["log_rate"] = np.log1p(df["Rate"].clip(lower=0))
+    df["log_totbytes"] = np.log1p(df["TotBytes"].clip(lower=0))
+    df["log_totpkts"] = np.log1p(df["TotPkts"].clip(lower=0))
+
+    return df
\ No newline at end of file
diff --git a/modeling/src/preprocessing.py b/modeling/src/preprocessing.py
new file mode 100644
index 0000000..c596c9d
--- /dev/null
+++ b/modeling/src/preprocessing.py
@@ -0,0 +1,80 @@
+import sys
+from pathlib import Path
+from .feature_engineering import add_engineered_features
+
+REPO_ROOT = Path(__file__).resolve().parent.parent.parent
+sys.path.append(str(REPO_ROOT))
+
+import pandas as pd
+from contracts.schemas import (
+    DROPPED_COLUMNS,
+    FEATURE_SCHEMA,
+    NULL_SENTINEL,
+)
+
+
+def load_clean_csv(csv_path: str, engineer: bool = True) -> pd.DataFrame:
+    """Load Combined.csv and apply the contract preprocessing.
+
+    Args:
+        csv_path: path to Combined.csv
+        engineer: if True, add engineered features from feature_engineering.py
+    """
+    df = pd.read_csv(csv_path, low_memory=False)
+
+    df = df.drop(columns=list(DROPPED_COLUMNS.keys()))
+
+    df["is_tcp"] = (df["Proto"] == "tcp").astype(int)
+    df["has_dst_reply"] = df["dTtl"].notna().astype(int)
+
+    numeric_cols = [f.name for f in FEATURE_SCHEMA if f.dtype in ("float", "int")]
+    df[numeric_cols] = df[numeric_cols].fillna(NULL_SENTINEL)
+
+    categorical_cols = [f.name for f in FEATURE_SCHEMA if f.dtype == "category"]
+    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
+
+    if engineer:
+        df = add_engineered_features(df)
+
+    return df
+
+
+DEFAULT_CACHE_PATH = REPO_ROOT / "modeling" / "artifacts" / "processed" / "clean.parquet"
+DEFAULT_CSV_PATH = REPO_ROOT / "data" / "raw" / "Combined.csv"
+
+
+def load_or_build(
+    csv_path: str,
+    cache_path: str = None,
+    engineer: bool = True,
+) -> pd.DataFrame:
+    """Load preprocessed data from cache if available, else build and cache."""
+    if cache_path is None:
+        cache_path = (
+            "modeling/artifacts/processed/clean_engineered.parquet"
+            if engineer
+            else "modeling/artifacts/processed/clean.parquet"
+        )
+
+    cache = Path(cache_path)
+
+    if cache.exists():
+        print(f"Loading cached preprocessed data from {cache}")
+        return pd.read_parquet(cache)
+
+    print(f"No cache found at {cache} — running preprocessing...")
+    df = load_clean_csv(csv_path, engineer=engineer)
+
+    cache.parent.mkdir(parents=True, exist_ok=True)
+    df.to_parquet(cache)
+    print(f"Cached to {cache}")
+    return df
+
+
+if __name__ == "__main__":
+    df = load_or_build(DEFAULT_CSV_PATH)
+    print(f"Shape: {df.shape}")
+    print(
+        f"Any nulls left in numeric features: "
+        f"{df.select_dtypes('number').isnull().sum().sum()}"
+    )
diff --git a/modeling/src/register_model.py b/modeling/src/register_model.py
new file mode 100644
index 0000000..6ed138c
--- /dev/null
+++ b/modeling/src/register_model.py
@@ -0,0 +1,83 @@
+"""
+Register the production model in the MLflow Model Registry.
+
+This script promotes a specific run's logged model to a named, versioned
+entry in the registry, then assigns aliases (staging, production) that
+downstream code (FastAPI serving, drift monitoring) will use to load the
+model without hardcoding a version.
+
+Run this ONCE after you've picked a winning run. Re-running creates a new
+version, which is what you want for genuine model updates but not for
+re-registration of the same model.
+"""
+
+import sys
+from pathlib import Path
+sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
+
+import mlflow
+from mlflow.tracking import MlflowClient
+
+
+MODEL_NAME = "5g-nidd-attack-classifier"
+
+WINNING_RUN_ID = "7742eed4fa9d45a1a43d0d09f686e1db"
+
+
+def register():
+    client = MlflowClient()
+
+    # Register the model artifact from that run
+    model_uri = f"runs:/{WINNING_RUN_ID}/model"
+    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
+    print(f"Registered {MODEL_NAME} as version {result.version}")
+
+    # Set the "staging" alias to point at this new version
+    client.set_registered_model_alias(
+        name=MODEL_NAME,
+        alias="staging",
+        version=result.version,
+    )
+    print(f"Alias 'staging' now points at version {result.version}")
+
+    # Add a description explaining what this model is
+    client.update_registered_model(
+        name=MODEL_NAME,
+        description=(
+            "XGBoost multiclass classifier for 5G-NIDD attack detection. "
+            "9 classes: Benign + 8 attack types. Trained on 70/15/15 stratified "
+            "split of Combined.csv (schema v1.0.1, Offset dropped). "
+            "77 features, sample_weight='balanced' for class imbalance. "
+            "Achieves macro-F1 0.92 with near-perfect attack recall (98-100%) "
+            "and Benign recall 0.41 (documented false-positive trade-off, "
+            "not addressable within per-flow feature representation)."
+        ),
+    )
+
+    # Description on the specific version — what makes THIS version distinct
+    client.update_model_version(
+        name=MODEL_NAME,
+        version=result.version,
+        description=(
+            f"Initial production candidate. Trained from run {WINNING_RUN_ID}. "
+            "Features: 77 baseline features (52 raw + is_tcp + has_dst_reply, "
+            "after 9 dropped columns per schema v1.0.1). "
+            "Engineered features tested separately, no measurable improvement, "
+            "not included in production pipeline (see notes.md). "
+            "Model: XGBClassifier, tree_method='hist', "
+            "sample_weight='balanced', n_estimators=200, max_depth=6."
+        ),
+    )
+
+    print(f"\nRegistered model summary:")
+    print(f"  Name:       {MODEL_NAME}")
+    print(f"  Version:    {result.version}")
+    print(f"  Alias:      staging")
+    print(f"  Load with:  mlflow.xgboost.load_model('models:/{MODEL_NAME}@staging')")
+
+
+if __name__ == "__main__":
+    if WINNING_RUN_ID == "PASTE_YOUR_RUN_ID_HERE":
+        print("Error: paste your winning run_id into WINNING_RUN_ID first.")
+        sys.exit(1)
+    register()
\ No newline at end of file
diff --git a/modeling/src/splits.py b/modeling/src/splits.py
new file mode 100644
index 0000000..d2062a3
--- /dev/null
+++ b/modeling/src/splits.py
@@ -0,0 +1,65 @@
+from pathlib import Path
+import pandas as pd
+from sklearn.model_selection import train_test_split
+
+LABEL_COLS = ["Label", "Attack Type", "Attack Tool"]
+
+RANDOM_STATE = 42
+
+def make_splits(df:pd.DataFrame, target='Attack Type'):
+    """Return X_train, X_val, X_test, y_train, y_val, y_test.
+
+    70/15/15 stratified split. Uses RANDOM_STATE = 42 for reproducibility.
+    """
+    
+    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
+    X = df[feature_cols]
+    y = df[target]
+
+    # First split: 70% train, 30% temp (val + test)
+    X_train, X_temp, y_train, y_temp = train_test_split(
+        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
+    )
+
+    # Second split: halve the 30% into 15% val, 15% test
+    X_val, X_test, y_val, y_test = train_test_split(
+        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
+    )
+
+    return X_train, X_val, X_test, y_train, y_val, y_test
+
+def save_splits(splits: tuple, out_dir: str = "modeling/artifacts/splits"):
+    """Save all six arrays as parquet files for fast reload."""
+    out = Path(out_dir)
+    out.mkdir(parents=True, exist_ok=True)
+
+    X_train, X_val, X_test, y_train, y_val, y_test = splits
+    X_train.to_parquet(out / "X_train.parquet")
+    X_val.to_parquet(out / "X_val.parquet")
+    X_test.to_parquet(out / "X_test.parquet")
+    y_train.to_frame().to_parquet(out / "y_train.parquet")
+    y_val.to_frame().to_parquet(out / "y_val.parquet")
+    y_test.to_frame().to_parquet(out / "y_test.parquet")
+
+    print(f"Splits saved to {out}/")
+    
+if __name__ == "__main__":
+    import sys
+    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
+    from modeling.src.preprocessing import load_or_build
+
+    df = load_or_build("data/raw/Combined.csv", engineer=False)
+    splits = make_splits(df)
+    X_train, X_val, X_test, y_train, y_val, y_test = splits
+
+    print(f"Train: {X_train.shape}")
+    print(f"Val:   {X_val.shape}")
+    print(f"Test:  {X_test.shape}")
+    print("\nTest set class distribution (verify stratification):")
+    print(y_test.value_counts())
+
+    save_splits(splits)
+    
+    
+    
+    
\ No newline at end of file
diff --git a/modeling/src/train.py b/modeling/src/train.py
new file mode 100644
index 0000000..75937d1
--- /dev/null
+++ b/modeling/src/train.py
@@ -0,0 +1,686 @@
+import sys
+from pathlib import Path
+
+sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
+
+import pandas as pd
+import numpy as np
+import mlflow
+import mlflow.sklearn
+from sklearn.linear_model import LogisticRegression
+from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score
+from sklearn.pipeline import Pipeline
+from sklearn.preprocessing import StandardScaler
+import matplotlib.pyplot as plt
+from sklearn.metrics import ConfusionMatrixDisplay
+from sklearn.ensemble import RandomForestClassifier
+from xgboost import XGBClassifier
+from sklearn.preprocessing import LabelEncoder
+from sklearn.utils.class_weight import compute_sample_weight
+import mlflow.xgboost
+
+
+REPO_ROOT = Path(__file__).resolve().parent.parent.parent
+SPLITS_DIR = REPO_ROOT / "modeling" / "artifacts" / "splits"
+
+EXPERIMENT_NAME = "5g-nidd-attack-classification"
+
+
+def load_splits():
+    """Load the parquet splits produced by splits.py."""
+    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
+    X_val = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
+    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()
+    y_val = pd.read_parquet(SPLITS_DIR / "y_val.parquet").squeeze()
+    return X_train, X_val, y_train, y_val
+
+
+def train_logreg():
+    X_train, X_val, y_train, y_val = load_splits()
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+
+    with mlflow.start_run(run_name="baseline_logreg_scaled"):
+        # Log hyperparameters
+        params = {
+            "model": "LogisticRegression",
+            "scaler": "StandardScaler",
+            "class_weight": "balanced",
+            "max_iter": 1000,
+            "solver": "lbfgs",
+            "random_state": 42,
+        }
+        mlflow.log_params(params)
+
+        # Fit
+        model = Pipeline(
+            [
+                ("scaler", StandardScaler()),
+                (
+                    "logreg",
+                    LogisticRegression(
+                        class_weight="balanced",
+                        max_iter=1000,
+                        solver="lbfgs",
+                        random_state=42,
+                        n_jobs=-1,
+                    ),
+                ),
+            ]
+        )
+        model.fit(X_train, y_train)
+
+        # Evaluate on val
+        y_pred = model.predict(X_val)
+        report = classification_report(y_val, y_pred, output_dict=True)
+
+        # Log overall + per-class metrics
+        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
+        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
+        mlflow.log_metric("accuracy", report["accuracy"])
+        for class_name, metrics in report.items():
+            if isinstance(metrics, dict) and class_name not in (
+                "macro avg",
+                "weighted avg",
+            ):
+                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
+                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
+
+        # Log the model itself
+        mlflow.sklearn.log_model(model, name="model")
+
+        # Log confusion matrix
+        fig, ax = plt.subplots(figsize=(10, 8))
+        ConfusionMatrixDisplay.from_predictions(
+            y_val, y_pred, ax=ax, xticks_rotation=45
+        )
+        plt.tight_layout()
+        plt.savefig("confusion_matrix.png")
+        mlflow.log_artifact("confusion_matrix.png")
+        plt.close()
+
+        # Feature importance
+        logreg = model.named_steps["logreg"]
+        importance = np.abs(logreg.coef_).mean(axis=0)
+        top20 = (
+            pd.Series(importance, index=X_train.columns)
+            .sort_values(ascending=False)
+            .head(20)
+        )
+
+        fig, ax = plt.subplots(figsize=(8, 6))
+        top20.plot.barh(ax=ax)
+        ax.set_title("Top 20 feature importances (mean |coef|)")
+        plt.tight_layout()
+        plt.savefig("feature_importance.png")
+        mlflow.log_artifact("feature_importance.png")
+        plt.close()
+
+        # Print for immediate feedback
+        print(classification_report(y_val, y_pred))
+        print("\nTop 10 features by importance:")
+        print(top20.head(10))
+
+
+def train_logreg_no_toolprint():
+    """Train LogReg without sMeanPktSz to test robustness against tool fingerprinting."""
+    X_train, X_val, y_train, y_val = load_splits()
+    X_train = X_train.drop(columns=["sMeanPktSz"])
+    X_val = X_val.drop(columns=["sMeanPktSz"])
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+    with mlflow.start_run(run_name="logreg_scaled_no_sMeanPktSz"):
+        # Log hyperparameters
+        params = {
+            "model": "LogisticRegression",
+            "scaler": "StandardScaler",
+            "dropped_feature": "sMeanPktSz",
+            "class_weight": "balanced",
+            "max_iter": 5000,
+            "solver": "lbfgs",
+            "random_state": 42,
+        }
+        mlflow.log_params(params)
+
+        # Fit
+        model = Pipeline(
+            [
+                ("scaler", StandardScaler()),
+                (
+                    "logreg",
+                    LogisticRegression(
+                        class_weight="balanced",
+                        max_iter=5000,
+                        solver="lbfgs",
+                        random_state=42,
+                        n_jobs=-1,
+                    ),
+                ),
+            ]
+        )
+        model.fit(X_train, y_train)
+
+        # Evaluate on val
+        y_pred = model.predict(X_val)
+        report = classification_report(y_val, y_pred, output_dict=True)
+
+        # Log overall + per-class metrics
+        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
+        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
+        mlflow.log_metric("accuracy", report["accuracy"])
+        for class_name, metrics in report.items():
+            if isinstance(metrics, dict) and class_name not in (
+                "macro avg",
+                "weighted avg",
+            ):
+                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
+                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
+
+        # Log the model itself
+        mlflow.sklearn.log_model(model, name="model")
+
+        # Log confusion matrix
+        fig, ax = plt.subplots(figsize=(10, 8))
+        ConfusionMatrixDisplay.from_predictions(
+            y_val, y_pred, ax=ax, xticks_rotation=45
+        )
+        plt.tight_layout()
+        plt.savefig("confusion_matrix.png")
+        mlflow.log_artifact("confusion_matrix.png")
+        plt.close()
+
+        # Feature importance
+        logreg = model.named_steps["logreg"]
+        importance = np.abs(logreg.coef_).mean(axis=0)
+        top20 = (
+            pd.Series(importance, index=X_train.columns)
+            .sort_values(ascending=False)
+            .head(20)
+        )
+
+        fig, ax = plt.subplots(figsize=(8, 6))
+        top20.plot.barh(ax=ax)
+        ax.set_title("Top 20 feature importances (mean |coef|)")
+        plt.tight_layout()
+        plt.savefig("feature_importance.png")
+        mlflow.log_artifact("feature_importance.png")
+        plt.close()
+
+        # Print for immediate feedback
+        print(classification_report(y_val, y_pred))
+        print("\nTop 10 features by importance:")
+        print(top20.head(10))
+
+
+
+
+def train_random_forest():
+    X_train, X_val, y_train, y_val = load_splits()
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+
+    with mlflow.start_run(run_name="random_forest_baseline"):
+        params = {
+            "model": "RandomForestClassifier",
+            "n_estimators": 100,
+            "class_weight": "balanced",
+            "max_depth": None,
+            "n_jobs": -1,
+            "random_state": 42,
+        }
+        mlflow.log_params(params)
+
+        model = RandomForestClassifier(
+            n_estimators=100,
+            class_weight="balanced",
+            n_jobs=-1,
+            random_state=42,
+        )
+        model.fit(X_train, y_train)
+
+        y_pred = model.predict(X_val)
+        report = classification_report(y_val, y_pred, output_dict=True)
+
+        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
+        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
+        mlflow.log_metric("accuracy", report["accuracy"])
+        for class_name, metrics in report.items():
+            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
+                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
+                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
+
+        mlflow.sklearn.log_model(model, name="model")
+
+        # Feature importance — RF gives this natively, no coef math needed
+        importance = pd.Series(
+            model.feature_importances_, index=X_train.columns
+        ).sort_values(ascending=False)
+
+        # ... same feature_importance + confusion_matrix logging blocks as before ...
+        
+        # Log confusion matrix
+        fig, ax = plt.subplots(figsize=(10, 8))
+        ConfusionMatrixDisplay.from_predictions(
+            y_val, y_pred, ax=ax, xticks_rotation=45
+        )
+        plt.tight_layout()
+        plt.savefig("confusion_matrix.png")
+        mlflow.log_artifact("confusion_matrix.png")
+        plt.close()
+        
+        
+        # Feature importance
+        top20 = importance.head(20)
+        fig, ax = plt.subplots(figsize=(8, 6))
+        top20.plot.barh(ax=ax)
+        ax.set_title("Top 20 feature importances (mean |coef|)")
+        plt.tight_layout()
+        plt.savefig("feature_importance.png")
+        mlflow.log_artifact("feature_importance.png")
+        plt.close()
+        
+        
+        print(classification_report(y_val, y_pred))
+        print("\nTop 10 features by importance:")
+        print(top20.head(10))
+        
+def train_xgboost():
+    X_train, X_val, y_train, y_val = load_splits()
+
+    # XGBoost needs integer labels
+    le = LabelEncoder()
+    y_train_enc = le.fit_transform(y_train)
+    y_val_enc = le.transform(y_val)
+
+    # Handle class imbalance via sample weights (XGBoost's equivalent to class_weight='balanced')
+    sample_weights = compute_sample_weight("balanced", y_train_enc)
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+
+    with mlflow.start_run(run_name="xgboost_engineered_features"):
+        params = {
+            "model": "XGBClassifier",
+            "n_estimators": 200,
+            "max_depth": 6,
+            "learning_rate": 0.1,
+            "class_weighting": "balanced_sample_weights",
+            "n_jobs": -1,
+            "random_state": 42,
+            "features": "baseline + engineered (ratios + logs)"
+        }
+        mlflow.log_params(params)
+
+        model = XGBClassifier(
+            n_estimators=200,
+            max_depth=6,
+            learning_rate=0.1,
+            n_jobs=-1,
+            random_state=42,
+            tree_method="hist",   # fast histogram-based training
+        )
+        model.fit(X_train, y_train_enc, sample_weight=sample_weights)
+
+        y_pred_enc = model.predict(X_val)
+        y_pred = le.inverse_transform(y_pred_enc)
+
+        report = classification_report(y_val, y_pred, output_dict=True)
+        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
+        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
+        mlflow.log_metric("accuracy", report["accuracy"])
+        for class_name, metrics in report.items():
+            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
+                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
+                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
+
+        mlflow.xgboost.log_model(model, name="model")
+
+        importance = pd.Series(
+            model.feature_importances_, index=X_train.columns
+        ).sort_values(ascending=False)
+
+        # ... same confusion matrix + feature importance logging as before ...
+        # Log confusion matrix
+        fig, ax = plt.subplots(figsize=(10, 8))
+        ConfusionMatrixDisplay.from_predictions(
+            y_val, y_pred, ax=ax, xticks_rotation=45
+        )
+        plt.tight_layout()
+        plt.savefig("confusion_matrix.png")
+        mlflow.log_artifact("confusion_matrix.png")
+        plt.close()
+        
+        
+        # Feature importance
+        top20 = importance.head(20)
+        fig, ax = plt.subplots(figsize=(8, 6))
+        top20.plot.barh(ax=ax)
+        ax.set_title("Top 20 feature importances (mean |coef|)")
+        plt.tight_layout()
+        plt.savefig("feature_importance.png")
+        mlflow.log_artifact("feature_importance.png")
+        plt.close()
+        
+
+        print(classification_report(y_val, y_pred))
+        print("\nTop 10 features by importance:")
+        print(top20.head(10))
+
+
+def threshold_analysis(run_id: str):
+    """Sweep UDPFlood threshold on a previously-trained model loaded from MLflow."""
+    X_train, X_val, y_train, y_val = load_splits()
+
+    # Load the trained model from MLflow
+    model_uri = f"runs:/{run_id}/model"
+    model = mlflow.xgboost.load_model(model_uri)
+    print(f"Loaded model from {model_uri}")
+
+    # The model expects encoded labels — refit the encoder on training labels
+    # (deterministic given the same y_train, so this reproduces the training-time encoding)
+    le = LabelEncoder()
+    le.fit(y_train)
+
+    y_proba = model.predict_proba(X_val)
+    udpflood_idx = list(le.classes_).index("UDPFlood")
+
+    thresholds = np.arange(0.30, 0.96, 0.05)
+    results = []
+
+    for t in thresholds:
+        udpflood_prob = y_proba[:, udpflood_idx]
+        default_preds = np.argmax(y_proba, axis=1)
+        alt_preds = np.argsort(y_proba, axis=1)[:, -2]
+
+        # Only override when the model was going to predict UDPFlood AND prob is below threshold
+        override_mask = (default_preds == udpflood_idx) & (udpflood_prob <= t)
+        preds_enc = np.where(override_mask, alt_preds, default_preds)
+
+        y_pred = le.inverse_transform(preds_enc)
+
+        results.append({
+            "threshold": round(t, 2),
+            "benign_recall": recall_score(y_val, y_pred, labels=["Benign"], average="macro"),
+            "udpflood_recall": recall_score(y_val, y_pred, labels=["UDPFlood"], average="macro"),
+            "macro_f1": f1_score(y_val, y_pred, average="macro"),
+        })
+
+    results_df = pd.DataFrame(results)
+    print(results_df.to_string(index=False))
+
+    fig, ax = plt.subplots(figsize=(9, 6))
+    ax.plot(results_df["threshold"], results_df["benign_recall"], marker="o", label="Benign recall")
+    ax.plot(results_df["threshold"], results_df["udpflood_recall"], marker="o", label="UDPFlood recall")
+    ax.plot(results_df["threshold"], results_df["macro_f1"], marker="o", label="Macro-F1", linestyle="--")
+    ax.set_xlabel("UDPFlood prediction threshold")
+    ax.set_ylabel("Metric value")
+    ax.set_title("Trade-off: Benign vs UDPFlood recall under threshold tuning")
+    ax.legend()
+    ax.grid(alpha=0.3)
+    plt.tight_layout()
+    plt.savefig("threshold_analysis.png")
+    plt.close()
+    print("Saved threshold_analysis.png")
+
+    return results_df
+
+
+def train_xgboost_binary():
+    """Train a binary XGBoost: Benign vs Malicious.
+
+    Framed as an operational-first model: for a real 5G security system, the
+    primary question is often 'is this flow malicious?' rather than 'which
+    specific attack type is this?'. This model answers the binary question
+    directly, sidestepping the Benign↔UDPFlood multiclass confusion.
+    """
+    X_train, X_val, y_train, y_val = load_splits()
+
+    # Derive binary labels from Attack Type
+    y_train_bin = (y_train != "Benign").astype(int)  # 1 = malicious, 0 = benign
+    y_val_bin = (y_val != "Benign").astype(int)
+
+    print(f"Train class balance: malicious={y_train_bin.sum()}, benign={(1-y_train_bin).sum()}")
+    print(f"Val class balance:   malicious={y_val_bin.sum()}, benign={(1-y_val_bin).sum()}")
+
+    # scale_pos_weight is XGBoost's binary imbalance handling
+    n_neg = (y_train_bin == 0).sum()
+    n_pos = (y_train_bin == 1).sum()
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+
+    with mlflow.start_run(run_name="xgboost_binary"):
+        params = {
+            "model": "XGBClassifier",
+            "task": "binary",
+            "target": "Benign vs Malicious",
+            "n_estimators": 200,
+            "max_depth": 6,
+            "learning_rate": 0.1,
+            "n_jobs": -1,
+            "random_state": 42,
+            "tree_method": "hist",
+        }
+        mlflow.log_params(params)
+
+        model = XGBClassifier(
+            n_estimators=200,
+            max_depth=6,
+            learning_rate=0.1,
+            n_jobs=-1,
+            random_state=42,
+            tree_method="hist",
+        )
+        model.fit(X_train, y_train_bin)
+
+        y_pred = model.predict(X_val)
+        y_proba = model.predict_proba(X_val)[:, 1]
+
+        # Binary-specific metrics
+        from sklearn.metrics import (
+            classification_report, f1_score, precision_score,
+            recall_score, roc_auc_score
+        )
+
+        report = classification_report(
+            y_val_bin, y_pred,
+            target_names=["Benign", "Malicious"],
+            output_dict=True,
+        )
+
+        mlflow.log_metric("accuracy", report["accuracy"])
+        mlflow.log_metric("f1_binary", f1_score(y_val_bin, y_pred))
+        mlflow.log_metric("precision_malicious", precision_score(y_val_bin, y_pred))
+        mlflow.log_metric("recall_malicious", recall_score(y_val_bin, y_pred))
+        mlflow.log_metric("precision_benign", precision_score(1 - y_val_bin, 1 - y_pred))
+        mlflow.log_metric("recall_benign", recall_score(1 - y_val_bin, 1 - y_pred))
+        mlflow.log_metric("roc_auc", roc_auc_score(y_val_bin, y_proba))
+
+        mlflow.xgboost.log_model(model, name="model")
+
+        # Confusion matrix
+        from sklearn.metrics import ConfusionMatrixDisplay
+        fig, ax = plt.subplots(figsize=(6, 5))
+        ConfusionMatrixDisplay.from_predictions(
+            y_val_bin, y_pred,
+            display_labels=["Benign", "Malicious"],
+            ax=ax,
+        )
+        plt.tight_layout()
+        plt.savefig("confusion_matrix.png")
+        mlflow.log_artifact("confusion_matrix.png")
+        plt.close()
+
+        # Feature importance
+        importance = pd.Series(
+            model.feature_importances_, index=X_train.columns
+        ).sort_values(ascending=False).head(20)
+
+        fig, ax = plt.subplots(figsize=(8, 6))
+        importance.plot.barh(ax=ax)
+        ax.set_title("Top 20 feature importances (binary XGBoost)")
+        ax.invert_yaxis()
+        plt.tight_layout()
+        plt.savefig("feature_importance.png")
+        mlflow.log_artifact("feature_importance.png")
+        plt.close()
+
+        print("\n" + classification_report(
+            y_val_bin, y_pred,
+            target_names=["Benign", "Malicious"],
+        ))
+        print(f"\nROC-AUC: {roc_auc_score(y_val_bin, y_proba):.4f}")
+        print("\nTop 10 features:")
+        print(importance.head(10))
+
+def probe_benign_vs_udpflood():
+    """Quick probe: what's the best possible Benign vs UDPFlood classifier?
+    Not a production model — just measures the ceiling of the two-class problem
+    to decide whether a cascade approach is worth building.
+    """
+    X_train, X_val, y_train, y_val = load_splits()
+    X_train = X_train.drop(columns=["sMeanPktSz"])
+    X_val = X_val.drop(columns=["sMeanPktSz"])
+
+    # Subset: only Benign and UDPFlood
+    train_mask = y_train.isin(["Benign", "UDPFlood"])
+    val_mask = y_val.isin(["Benign", "UDPFlood"])
+
+    X_train_sub = X_train[train_mask]
+    y_train_sub = y_train[train_mask]
+    X_val_sub = X_val[val_mask]
+    y_val_sub = y_val[val_mask]
+
+    print(f"Train: {X_train_sub.shape}, balance: {y_train_sub.value_counts().to_dict()}")
+    print(f"Val:   {X_val_sub.shape}, balance: {y_val_sub.value_counts().to_dict()}")
+
+    # Binary encoding
+    y_train_bin = (y_train_sub == "UDPFlood").astype(int)
+    y_val_bin = (y_val_sub == "UDPFlood").astype(int)
+
+    from sklearn.utils.class_weight import compute_sample_weight
+    weights = compute_sample_weight("balanced", y_train_bin)
+
+    model = XGBClassifier(
+        n_estimators=200,
+        max_depth=6,
+        learning_rate=0.1,
+        n_jobs=-1,
+        random_state=42,
+        tree_method="hist",
+    )
+    model.fit(X_train_sub, y_train_bin, sample_weight=weights)
+
+    y_pred = model.predict(X_val_sub)
+    y_proba = model.predict_proba(X_val_sub)[:, 1]
+
+    from sklearn.metrics import (
+        classification_report, roc_auc_score, f1_score,
+    )
+    print("\nClassification report:")
+    print(classification_report(
+        y_val_bin, y_pred,
+        target_names=["Benign", "UDPFlood"],
+    ))
+    print(f"ROC-AUC: {roc_auc_score(y_val_bin, y_proba):.4f}")
+    print(f"F1: {f1_score(y_val_bin, y_pred):.4f}")
+
+    # Importance — where does this specialist focus?
+    importance = pd.Series(
+        model.feature_importances_, index=X_train.columns
+    ).sort_values(ascending=False).head(10)
+    print("\nTop 10 features:")
+    print(importance)
+
+
+def train_xgboost_manual_weights():
+    """XGBoost multiclass with manually tuned class weights.
+
+    The default sample_weight='balanced' weights each class inversely to its
+    frequency, which was found to over-correct for UDPFlood (causing the
+    Benign->UDPFlood false-positive tendency documented in prior runs). Here
+    we deliberately ease off the UDPFlood weight while preserving heavy
+    weighting for the truly rare attack classes (ICMPFlood especially).
+    """
+    X_train, X_val, y_train, y_val = load_splits()
+
+    # Encode labels for XGBoost
+    le = LabelEncoder()
+    y_train_enc = le.fit_transform(y_train)
+    y_val_enc = le.transform(y_val)
+
+    # Manual weights — reasoning per class:
+    #  Benign: slightly up-weighted vs balanced (we want to reduce Benign FPs)
+    #  UDPFlood: slightly down-weighted (was over-predicted with 'balanced')
+    #  ICMPFlood: heavy weight — rarest class, only 808 train rows
+    #  SYNFlood: elevated — moderate rare class
+    #  Scans (SYNScan, TCPConnectScan, UDPScan): elevated to keep recall high
+    #  HTTPFlood, SlowrateDoS: baseline (they were already fine at balanced)
+    class_weights = {
+        "Benign": 1.5,
+        "UDPFlood": 0.7,
+        "HTTPFlood": 1.0,
+        "ICMPFlood": 10.0,
+        "SYNFlood": 3.0,
+        "SYNScan": 3.0,
+        "SlowrateDoS": 1.0,
+        "TCPConnectScan": 3.0,
+        "UDPScan": 3.0,
+    }
+
+    # Map each training row's label to its weight
+    sample_weights = np.array([class_weights[cls] for cls in y_train])
+
+    mlflow.set_experiment(EXPERIMENT_NAME)
+
+    with mlflow.start_run(run_name="xgboost_manual_weights"):
+        params = {
+            "model": "XGBClassifier",
+            "n_estimators": 200,
+            "max_depth": 6,
+            "learning_rate": 0.1,
+            "class_weighting": "manual_dict",
+            "n_jobs": -1,
+            "random_state": 42,
+        }
+        mlflow.log_params(params)
+
+        # Log the actual weight dict as a param for future reference
+        for cls, w in class_weights.items():
+            mlflow.log_param(f"weight_{cls}", w)
+
+        model = XGBClassifier(
+            n_estimators=200,
+            max_depth=6,
+            learning_rate=0.1,
+            n_jobs=-1,
+            random_state=42,
+            tree_method="hist",
+        )
+        model.fit(X_train, y_train_enc, sample_weight=sample_weights)
+
+        y_pred_enc = model.predict(X_val)
+        y_pred = le.inverse_transform(y_pred_enc)
+
+        report = classification_report(y_val, y_pred, output_dict=True)
+        mlflow.log_metric("macro_f1", f1_score(y_val, y_pred, average="macro"))
+        mlflow.log_metric("weighted_f1", f1_score(y_val, y_pred, average="weighted"))
+        mlflow.log_metric("accuracy", report["accuracy"])
+        for class_name, metrics in report.items():
+            if isinstance(metrics, dict) and class_name not in ("macro avg", "weighted avg"):
+                mlflow.log_metric(f"recall_{class_name}", metrics["recall"])
+                mlflow.log_metric(f"f1_{class_name}", metrics["f1-score"])
+
+        mlflow.xgboost.log_model(model, name="model")
+
+        importance = pd.Series(
+            model.feature_importances_, index=X_train.columns
+        ).sort_values(ascending=False)
+
+        print(classification_report(y_val, y_pred))
+        print("\nTop 10 features:")
+        print(importance.head(10))
+
+
+if __name__ == "__main__":
+    #RUN_ID = "ff2b46e8ac4a4ff0b9a054a57743de6e"
+    #threshold_analysis(RUN_ID)
+    train_xgboost_manual_weights()
\ No newline at end of file
diff --git a/requirements.txt b/requirements.txt
index c69ebad..06be05b 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -2,4 +2,9 @@ pandas==2.2.2
 kafka-python==2.0.2
 pyspark==3.5.1
 delta-spark==3.2.0
-python-dotenv==1.0.1
\ No newline at end of file
+python-dotenv==1.0.1
+scikit-learn==1.7.2
+pyarrow
+mlflow
+xgboost
+shap
\ No newline at end of file
