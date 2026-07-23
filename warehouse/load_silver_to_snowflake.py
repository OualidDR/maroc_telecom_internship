"""
warehouse/load_silver_to_snowflake.py
========================================
Reads the Silver Delta table from MinIO and loads it into Snowflake's
SILVER_FLOWS table. Run warehouse/create_silver_flows.sql once in a
Snowflake worksheet before the first load.

Usage:
    python3 warehouse/load_silver_to_snowflake.py              # append
    python3 warehouse/load_silver_to_snowflake.py --truncate    # clean reload
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

SILVER_PATH = "s3://silver/flows"
TABLE_NAME = "SILVER_FLOWS"


def read_silver_full() -> pd.DataFrame:
    """Read the ENTIRE Silver Delta table (no sampling -- unlike the GX script,
    we want everything that's accumulated so far loaded into the warehouse)."""
    storage_options = {
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    print(f"Reading Silver from {SILVER_PATH} ...")
    dt = DeltaTable(SILVER_PATH, storage_options=storage_options)
    df = dt.to_pandas()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from Silver.")
    return df


def get_snowflake_connection():
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Missing required .env values: {missing}")
        print("Fill in SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER / SNOWFLAKE_PASSWORD in your .env first.")
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
    parser = argparse.ArgumentParser(description="Load Silver (MinIO) -> Snowflake")
    parser.add_argument("--truncate", action="store_true",
                         help="Truncate SILVER_FLOWS before loading (clean reload, avoids duplicate accumulation)")
    args = parser.parse_args()

    df = read_silver_full()
    conn = get_snowflake_connection()

    if args.truncate:
        print(f"Truncating {TABLE_NAME} before load (--truncate was passed)...")
        conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {TABLE_NAME}")

    # Snowflake stores unquoted identifiers as UPPERCASE by default (the DDL
    # in create_silver_flows.sql doesn't quote anything). Uppercasing the
    # DataFrame's column names here guarantees an exact match regardless of
    # the mixed case used in contracts/schemas.py (Dur, Proto, Label, ...).
    df.columns = [c.upper() for c in df.columns]

    print(f"Writing {len(df)} rows to Snowflake table {TABLE_NAME} ...")
    success, n_chunks, n_rows, _ = write_pandas(
        conn,
        df,
        TABLE_NAME,
        auto_create_table=False,  # table is created explicitly by create_silver_flows.sql
        quote_identifiers=True,
        use_logical_type=True,  # correct handling of timezone-aware ingestion_timestamp
    )

    print(f"success={success}  chunks={n_chunks}  rows_loaded={n_rows}")
    conn.close()


if __name__ == "__main__":
    main()
