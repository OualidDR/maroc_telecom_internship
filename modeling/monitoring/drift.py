"""
Drift detection for live traffic against the reference distribution.

Two comparisons:
1. Data drift: are incoming features statistically different from training?
2. Prediction drift: is the model's output distribution shifting?

Generates an HTML report per batch that can be viewed in a browser or
served through the API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from datetime import datetime

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

REFERENCE_PATH = REPO_ROOT / "modeling" / "artifacts" / "monitoring" / "reference.parquet"
REPORTS_DIR = REPO_ROOT / "modeling" / "artifacts" / "monitoring" / "reports"


def _make_dataset(df: pd.DataFrame) -> Dataset:
    """Wrap a DataFrame as an Evidently Dataset with prediction column."""
    schema = DataDefinition(
        numerical_columns=df.select_dtypes("number").columns.tolist(),
        categorical_columns=[
            c for c in df.select_dtypes(include=["object", "bool"]).columns
            if c != "prediction"
        ],
    )
    return Dataset.from_pandas(df, data_definition=schema)


def compute_drift(current: pd.DataFrame, save_html: bool = True) -> dict:
    """Compare a batch of current flows to the reference.

    Args:
        current: DataFrame with same columns as reference, plus 'prediction'
        save_html: whether to save the interactive HTML report

    Returns:
        dict with drift summary (overall_drift_detected, share_of_drifted_columns, etc.)
    """
    reference = pd.read_parquet(REFERENCE_PATH)

    # Align columns — current might be missing some the reference has
    common = [c for c in reference.columns if c in current.columns]
    reference = reference[common]
    current = current[common]

    ref_ds = _make_dataset(reference)
    cur_ds = _make_dataset(current)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    # Extract summary
    summary = result.dict()

    if save_html:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
        result.save_html(str(html_path))
        print(f"Drift report saved to {html_path}")

    return summary


def simulate_drift_detection_from_val():
    """Diagnostic: run drift detection against the val set as a sanity check.

    Since val is drawn from the same distribution as train, drift should be
    near zero. This verifies the drift detection pipeline works.
    """
    print("Simulating drift detection using val set as 'current' batch...")
    X_val = pd.read_parquet("modeling/artifacts/splits/X_val.parquet")
    y_val = pd.read_parquet("modeling/artifacts/splits/y_val.parquet").squeeze()

    current = X_val.copy()
    current["prediction"] = y_val.values
    current = current.sample(5000, random_state=1)

    summary = compute_drift(current, save_html=True)

    # Pull out the numbers we care about
    metrics = summary.get("metrics", [])
    for m in metrics:
        if "DriftedColumnsCount" in m.get("metric_id", ""):
            print(f"\nDrifted columns: {m['value']['count']} out of {m['value']['share']*100:.1f}%")

def simulate_real_drift():
    """Prove detection works by feeding it a deliberately drifted batch.

    Takes val and applies distortions (shift means, scale variances) to
    simulate a distribution change. Evidently should detect this as drift.
    """
    print("Simulating REAL drift (deliberately shifted distributions)...")
    X_val = pd.read_parquet("modeling/artifacts/splits/X_val.parquet")
    y_val = pd.read_parquet("modeling/artifacts/splits/y_val.parquet").squeeze()

    current = X_val.sample(5000, random_state=1).copy()

    # Shift numerical features by adding noise scaled to their std
    for col in current.select_dtypes("number").columns:
        std = current[col].std()
        if std > 0:
            current[col] = current[col] + std * 2.0  # shift by 2 std

    current["prediction"] = y_val.loc[current.index].values

    summary = compute_drift(current, save_html=True)
    print("Report saved. This should show significant drift.")


if __name__ == "__main__":
    # simulate_drift_detection_from_val()  # for reference
    simulate_real_drift()
