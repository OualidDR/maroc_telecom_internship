"""
modeling/src/export_drift.py
==============================
Compute per-feature data drift (current distribution vs the training reference)
and load it into GOLD_DRIFT for the Power BI "Explainability / Monitoring" page.

Metric: PSI (population stability index) + a Kolmogorov-Smirnov test per feature.
PSI thresholds (industry standard): <0.1 stable, 0.1-0.2 moderate, >0.2 drift.
A column is flagged 'drifted' when PSI > 0.2. The dataset-level verdict fires
when >50% of columns drift (same threshold as the Evidently config in notes.md).

Current vs reference: uses X_val as the "current" batch against
modeling/artifacts/monitoring/reference.parquet (the training reference). On this
dataset they're near-identical, so this should report LITTLE/NO drift -- the
correct, healthy result, and a working monitoring signal for the dashboard.

Run warehouse/create_gold_explainability.sql once first, then:
    python modeling/src/export_drift.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ks_2samp

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_PATH = REPO_ROOT / "modeling" / "artifacts" / "monitoring" / "reference.parquet"
CURRENT_PATH = REPO_ROOT / "modeling" / "artifacts" / "splits" / "X_val.parquet"
TABLE_NAME = "GOLD_DRIFT"
PSI_DRIFT_THRESHOLD = 0.2
DATASET_DRIFT_SHARE = 0.5


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from the reference."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature -> no measurable drift
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref) + 1e-6
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur) + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift() -> pd.DataFrame:
    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_parquet(CURRENT_PATH)

    cols = [c for c in reference.columns
            if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]
    print(f"Comparing {len(cols)} numeric features "
          f"(reference n={len(reference)}, current n={len(current)}) ...")

    records = []
    for c in cols:
        # astype(float): one-hot dummy columns are bool, which np.quantile/KS
        # can't subtract -- cast to float so 0/1 features work like any numeric.
        ref_v = reference[c].dropna().astype(float).to_numpy()
        cur_v = current[c].dropna().astype(float).to_numpy()
        if len(ref_v) == 0 or len(cur_v) == 0:
            continue
        p = psi(ref_v, cur_v)
        ks_stat, ks_p = ks_2samp(ref_v, cur_v)
        records.append({
            "FEATURE": c,
            "PSI": round(float(p), 6),
            "KS_STATISTIC": round(float(ks_stat), 6),
            "KS_P_VALUE": float(ks_p),
            "DRIFTED": int(p > PSI_DRIFT_THRESHOLD),
        })

    df = pd.DataFrame(records)
    share = float(df["DRIFTED"].mean()) if len(df) else 0.0
    df["SHARE_DRIFTED"] = round(share, 4)
    df["DATASET_DRIFT_DETECTED"] = int(share > DATASET_DRIFT_SHARE)
    df["REFERENCE_SIZE"] = len(reference)
    df["CURRENT_SIZE"] = len(current)
    df["COMPUTED_AT"] = datetime.now(timezone.utc)

    print(f"\nDrifted columns (PSI>{PSI_DRIFT_THRESHOLD}): "
          f"{df['DRIFTED'].sum()}/{len(df)}  ({share:.0%})")
    print(f"Dataset drift detected: {bool(df['DATASET_DRIFT_DETECTED'].iloc[0])}")
    print("\nTop 5 features by PSI:")
    print(df.nlargest(5, "PSI")[["FEATURE", "PSI", "KS_STATISTIC", "DRIFTED"]].to_string(index=False))
    return df


def load_to_snowflake(df: pd.DataFrame):
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"), user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "NIDD_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "NIDD_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"),
    )
    conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {TABLE_NAME}")
    ok, _, n, _ = write_pandas(conn, df, TABLE_NAME, auto_create_table=False,
                               quote_identifiers=True, use_logical_type=True)
    print(f"\n{TABLE_NAME}: success={ok} rows_loaded={n}")
    conn.close()


if __name__ == "__main__":
    load_to_snowflake(compute_drift())
