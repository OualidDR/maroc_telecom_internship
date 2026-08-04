"""
warehouse/load_gold_predictions_to_snowflake.py
==================================================
Loads gold/model_predictions and gold/security_alerts (written by
serving/spark_predict_and_gold.py) from MinIO into Snowflake.
Run warehouse/create_gold_predictions.sql once first.

Usage:
    python3 warehouse/load_gold_predictions_to_snowflake.py
    python3 warehouse/load_gold_predictions_to_snowflake.py --truncate
"""

import argparse
import os
import sys

import pandas as pd
import snowflake.connector
from deltalake import DeltaTable
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas

sys.path.append(".")

load_dotenv()

TABLES = {
    "s3://gold/model_predictions": "GOLD_MODEL_PREDICTIONS",
    "s3://gold/security_alerts": "GOLD_SECURITY_ALERTS",
}


def read_delta(path: str) -> pd.DataFrame:
    storage_options = {
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    print(f"Reading {path} ...")
    dt = DeltaTable(path, storage_options=storage_options)
    df = dt.to_pandas()
    print(f"  {len(df)} rows, {len(df.columns)} columns.")
    return df


def get_snowflake_connection():
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Missing required .env values: {missing}")
        sys.exit(1)
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "NIDD_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "NIDD_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    conn = get_snowflake_connection()

    for path, table_name in TABLES.items():
        try:
            df = read_delta(path)
        except Exception as e:
            print(f"  Skipping {path}: {e} (probably empty / not written yet -- fine in mock/early testing)")
            continue

        if df.empty:
            print(f"  {table_name}: no rows to load, skipping.")
            continue

        df.columns = [c.upper() for c in df.columns]

        if args.truncate:
            print(f"  Truncating {table_name}...")
            conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

        success, n_chunks, n_rows, _ = write_pandas(
            conn, df, table_name,
            auto_create_table=False, quote_identifiers=True, use_logical_type=True,
        )
        print(f"  {table_name}: success={success} rows_loaded={n_rows}")

    conn.close()


if __name__ == "__main__":
    main()