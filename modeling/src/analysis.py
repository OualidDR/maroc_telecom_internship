"""
Explainability analysis for the registered XGBoost model.

Uses SHAP (SHapley Additive exPlanations) with the exact TreeExplainer for
per-class global summaries and per-flow local explanations. These outputs
serve two purposes:

1. Report: per-class feature-importance plots that go beyond the aggregate
   feature_importances_ vector, showing which features drive each attack
   type distinctly.
2. Serving: per-flow explanations returned alongside predictions by the
   FastAPI layer (Phase 6), so an analyst reviewing a flagged flow can see
   why the model made the call.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


MODEL_URI = "models:/5g-nidd-attack-classifier@staging"
SPLITS_DIR = Path("modeling/artifacts/splits")
OUT_DIR = Path("modeling/artifacts/shap")


def load_model_and_data():
    """Load the registered model and the val set from parquet."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = mlflow.xgboost.load_model(MODEL_URI)
    X_val = pd.read_parquet(SPLITS_DIR / "X_val.parquet")
    y_val = pd.read_parquet(SPLITS_DIR / "y_val.parquet").squeeze()
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()

    # Refit encoder so we can map class indices back to names
    le = LabelEncoder().fit(y_train)
    return model, X_val, y_val, le


def compute_shap_values(model, X_sample):
    """Compute SHAP values using XGBoost's native pred_contribs.

    Bypasses shap.TreeExplainer's XGBoost parser, which fails on recent
    XGBoost multiclass models (per-class base_score returned as JSON array).
    XGBoost's built-in SHAP support always stays in sync with its own format.
    """
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_sample)

    # pred_contribs returns:
    # - Multiclass: shape (n_samples, n_classes, n_features + 1)
    # - Binary: shape (n_samples, n_features + 1)
    # The last feature column is the bias (base_score) contribution.
    contribs = booster.predict(dmatrix, pred_contribs=True)

    # Strip the bias column and reshape to the shap-library layout:
    # (n_samples, n_features, n_classes)
    shap_values = contribs[:, :, :-1].transpose(0, 2, 1)

    return shap_values

def plot_global_summary(shap_values, X_sample, le, out_dir: Path):
    """Save a per-class SHAP summary bar plot.

    Aggregates absolute SHAP values across the sample to show which features
    the model relies on most for each class.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # shap.summary_plot with plot_type='bar' handles multiclass natively
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        class_names=list(le.classes_),
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'shap_global_bar.png'}")


def plot_per_class_beeswarm(shap_values, X_sample, le, out_dir: Path):
    """Save a beeswarm per class — shows feature effect direction and magnitude.

    A beeswarm reveals not just 'this feature is important' but 'high values
    of this feature push predictions toward this class' — direction + effect
    size, which the bar plot doesn't show.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, class_name in enumerate(le.classes_):
        # shap_values here is a 3D array indexed [:, :, class_idx]
        class_shap = shap_values[:, :, i]

        fig = plt.figure()
        shap.summary_plot(
            class_shap,
            X_sample,
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP feature effects for class: {class_name}")
        plt.tight_layout()
        plt.savefig(
            out_dir / f"shap_beeswarm_{class_name}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Saved {len(le.classes_)} per-class beeswarm plots to {out_dir}")


def local_explanation(model, X_row: pd.DataFrame, le, top_k: int = 5):
    """Explain a single prediction using XGBoost's native pred_contribs."""
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_row)
    contribs = booster.predict(dmatrix, pred_contribs=True)

    # Shape: (1, n_classes, n_features + 1) — strip bias
    contribs = contribs[:, :, :-1]

    proba = model.predict_proba(X_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = le.classes_[pred_idx]

    # SHAP values for the predicted class only
    class_shap = contribs[0, pred_idx, :]

    feature_names = X_row.columns
    top_k_idx = np.argsort(np.abs(class_shap))[-top_k:][::-1]

    top_features = [
        {
            "feature": feature_names[i],
            "value": float(X_row.iloc[0, i]),
            "shap_contribution": float(class_shap[i]),
        }
        for i in top_k_idx
    ]

    return {
        "predicted_class": pred_class,
        "probability": float(proba[pred_idx]),
        "top_features": top_features,
    }

def main():
    print("Loading model and data...")
    model, X_val, y_val, le = load_model_and_data()

    # SHAP on full val (182K rows) would take an hour+. Sample for global plots.
    print("Sampling 2000 rows for global SHAP...")
    sample_idx = X_val.sample(2000, random_state=42).index
    X_sample = X_val.loc[sample_idx]
    y_sample = y_val.loc[sample_idx]

    print("Computing SHAP values (may take 1-2 min)...")
    shap_values = compute_shap_values(model, X_sample)
    print(f"SHAP values shape: {shap_values.shape}")

    print("\nGenerating global summary bar plot...")
    plot_global_summary(shap_values, X_sample, le, OUT_DIR)

    print("Generating per-class beeswarm plots...")
    plot_per_class_beeswarm(shap_values, X_sample, le, OUT_DIR)

    print("\nDemo local explanation on 3 random val flows...")
    for i, idx in enumerate(X_val.sample(3, random_state=1).index):
        X_row = X_val.loc[[idx]]
        true_label = y_val.loc[idx]
        explanation = local_explanation(model, X_row, le, top_k=5)

        print(f"\n--- Flow #{i+1} ---")
        print(f"  True label:      {true_label}")
        print(f"  Predicted:       {explanation['predicted_class']} "
              f"(prob {explanation['probability']:.3f})")
        print(f"  Top features pushing toward predicted class:")
        for f in explanation['top_features']:
            direction = "↑" if f['shap_contribution'] > 0 else "↓"
            print(f"    {direction} {f['feature']:25s} "
                  f"value={f['value']:>12.4f}  shap={f['shap_contribution']:+.4f}")


if __name__ == "__main__":
    main()