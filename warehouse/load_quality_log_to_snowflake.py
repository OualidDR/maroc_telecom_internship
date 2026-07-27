"""
warehouse/load_quality_log_to_snowflake.py
=============================================
Reads the Great Expectations quality log (written by quality/gx_spark_validator.py
during streaming) from MinIO and loads it into Snowflake's QUALITY_LOG table.
Run warehouse/create_quality_log.sql once first.

Usage:
    python3 warehouse/load_quality_log_to_snowflake.py
    python3 warehouse/load_quality_log_to_snowflake.py --truncate
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

QUALITY_LOG_PATH = "s3://silver/_quality_log/gx_results"
TABLE_NAME = "QUALITY_LOG"


def read_quality_log() -> pd.DataFrame:
    storage_options = {
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }
    print(f"Reading quality log from {QUALITY_LOG_PATH} ...")
    dt = DeltaTable(QUALITY_LOG_PATH, storage_options=storage_options)
    df = dt.to_pandas()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
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
    parser = argparse.ArgumentParser(description="Load quality log (MinIO) -> Snowflake")
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    df = read_quality_log()
    df.columns = [c.upper() for c in df.columns]  # match Snowflake's default uppercase storage

    conn = get_snowflake_connection()

    if args.truncate:
        print(f"Truncating {TABLE_NAME} before load...")
        conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {TABLE_NAME}")

    print(f"Writing {len(df)} rows to Snowflake table {TABLE_NAME} ...")
    success, n_chunks, n_rows, _ = write_pandas(
        conn,
        df,
        TABLE_NAME,
        auto_create_table=False,
        quote_identifiers=True,
        use_logical_type=True,
    )
    print(f"success={success}  chunks={n_chunks}  rows_loaded={n_rows}")
    conn.close()


if __name__ == "__main__":
    main()
