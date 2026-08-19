"""
smoke_test_predict_raw.py
==========================
End-to-end smoke test for the /predict/raw endpoint.

Sends REAL raw flows (with their true Attack Type labels held aside) to the
running API and checks the predictions are sane. The point is to catch the
*silent* failure mode of /predict/raw: if column casing or preprocessing is
wrong, `reindex` fills every feature with 0, the model returns a confident
answer with NO error, and every row collapses to the same class. A test that
only checks "HTTP 200" would miss that -- so this one compares predictions to
the known labels and asserts the output actually varies.

Prereqs:
    - The API is running:  uvicorn modeling.serving.api:app --port 8000
    - The model is loaded (MLflow server up + model registered @staging).

Usage:
    python smoke_test_predict_raw.py                       # from data/raw/Combined.csv (default)
    python smoke_test_predict_raw.py --source silver       # from the Silver Delta table on MinIO
    python smoke_test_predict_raw.py --n-per-class 5        # more rows per attack type
    python smoke_test_predict_raw.py --url http://localhost:8000/predict/raw
"""

import argparse
import math
import os
import sys
import uuid

import pandas as pd
import requests

sys.path.append(".")
from contracts.schemas import FEATURE_COLUMNS  # noqa: E402

CSV_PATH = "data/raw/Combined.csv"
SILVER_PATH = "s3://silver/flows"
DEFAULT_URL = "http://localhost:8000/predict/raw"


def _sample_per_class(df: pd.DataFrame, label_col: str, n_per_class: int) -> pd.DataFrame:
    """Take up to n_per_class rows from each label, then shuffle the mix."""
    parts = [
        g.sample(min(n_per_class, len(g)), random_state=42)
        for _, g in df.groupby(label_col)
    ]
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def load_from_csv(n_per_class: int):
    print(f"Loading raw rows from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    sampled = _sample_per_class(df, "Attack Type", n_per_class)
    truth = sampled["Attack Type"].tolist()
    feats = sampled[[c for c in FEATURE_COLUMNS if c in sampled.columns]].copy()
    return feats, truth


def load_from_silver(n_per_class: int):
    from deltalake import DeltaTable

    storage_options = {
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    print(f"Loading raw rows from Silver Delta {SILVER_PATH} ...")
    df = DeltaTable(SILVER_PATH, storage_options=storage_options).to_pandas()
    # Silver renamed "Attack Type" -> attack_type (Delta forbids spaces).
    sampled = _sample_per_class(df, "attack_type", n_per_class)
    truth = sampled["attack_type"].tolist()
    feats = sampled[[c for c in FEATURE_COLUMNS if c in sampled.columns]].copy()
    return feats, truth


def _json_safe(v):
    """Make one value JSON-transportable: numpy scalar -> native Python, and
    NaN / inf -> None (JSON has no NaN; requests rejects it with allow_nan=False).
    NOTE: doing this per-value is required -- `df.where(notnull, None)` does NOT
    work on float columns, because pandas re-coerces the None back to NaN to keep
    the column float64.
    """
    if hasattr(v, "item"):          # numpy scalar -> python scalar
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):  # NaN or inf
        return None
    return v


def to_json_records(feats: pd.DataFrame) -> list[dict]:
    """One JSON-safe dict per row, with a unique event_id added."""
    records = []
    for i, (_, row) in enumerate(feats.iterrows()):
        rec = {k: _json_safe(v) for k, v in row.to_dict().items()}
        rec["event_id"] = f"smoke_{i:03d}_{uuid.uuid4().hex[:6]}"
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="Smoke test for /predict/raw")
    parser.add_argument("--source", choices=["csv", "silver"], default="csv",
                        help="Where to pull real raw flows from (default: csv)")
    parser.add_argument("--n-per-class", type=int, default=3,
                        help="Rows to sample per attack type (default: 3)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL (default: {DEFAULT_URL})")
    args = parser.parse_args()

    if args.source == "silver":
        from dotenv import load_dotenv
        load_dotenv()
        feats, truth = load_from_silver(args.n_per_class)
    else:
        feats, truth = load_from_csv(args.n_per_class)

    records = to_json_records(feats)
    print(f"Sending {len(records)} raw flows ({feats.shape[1]} raw feature cols) to {args.url}\n")

    try:
        resp = requests.post(args.url, json={"flows": records}, timeout=30)
    except requests.ConnectionError:
        print(f"[FAIL] Could not connect to {args.url}. Is the API running?")
        print("       Start it with:  uvicorn modeling.serving.api:app --port 8000")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    body = resp.json()
    preds = body.get("predictions", [])

    # --- Print the side-by-side comparison -----------------------------------
    print(f"{'event_id':<20} {'true_label':<16} {'predicted':<16} {'prob':>6}  ok")
    print("-" * 68)
    n_match = 0
    n_attack = n_attack_match = 0
    for t, p in zip(truth, preds):
        pred = p["predicted_class"]
        ok = (pred == t)
        n_match += ok
        if t != "Benign":
            n_attack += 1
            n_attack_match += ok
        print(f"{p['event_id']:<20} {t:<16} {pred:<16} {p['probability']:>6.2f}  {'Y' if ok else '.'}")

    distinct_preds = {p["predicted_class"] for p in preds}

    # --- Verdicts ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SMOKE-TEST VERDICTS")
    print("=" * 68)

    # 1. Count returned
    ok_count = len(preds) == len(records)
    print(f"[{'PASS' if ok_count else 'FAIL'}] returned {len(preds)}/{len(records)} predictions")

    # 2. event_id passthrough
    ok_ids = all(p.get("event_id") for p in preds)
    print(f"[{'PASS' if ok_ids else 'FAIL'}] every prediction carries its event_id")

    # 3. NOT collapsed to one class -- the silent-zeros symptom
    ok_variety = len(distinct_preds) > 1
    print(f"[{'PASS' if ok_variety else 'FAIL'}] predictions span {len(distinct_preds)} classes "
          f"(collapse to 1 class ⇒ preprocessing/casing bug)")

    # 4. Attack-class accuracy. Benign is EXCLUDED on purpose: the documented
    #    Benign->UDPFlood confusion (notes.md) means Benign is expected to be
    #    misclassified often, so it's a bad accuracy signal. Attack rows should
    #    score ~98-100% recall if preprocessing is correct.
    if n_attack:
        acc = n_attack_match / n_attack
        ok_acc = acc >= 0.80
        print(f"[{'PASS' if ok_acc else 'FAIL'}] attack-class accuracy {acc:.0%} "
              f"({n_attack_match}/{n_attack}, Benign excluded; expect ≥80%)")
    else:
        ok_acc = True
        print("[WARN] no attack rows sampled -- can't check attack accuracy")

    # Benign is informational only, never a fail.
    n_benign = truth.count("Benign")
    if n_benign:
        benign_as_udp = sum(
            1 for t, p in zip(truth, preds)
            if t == "Benign" and p["predicted_class"] == "UDPFlood"
        )
        print(f"[INFO] Benign rows: {n_benign}; predicted UDPFlood: {benign_as_udp} "
              f"(expected per documented Benign↔UDPFlood confusion)")

    all_ok = ok_count and ok_ids and ok_variety and ok_acc
    print("\n" + ("✅ SMOKE TEST PASSED — /predict/raw looks healthy."
                  if all_ok else
                  "❌ SMOKE TEST FAILED — see the FAIL line(s) above."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
