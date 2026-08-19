"""
warehouse/run_gold_ddl.py
===========================
Runs create_gold_predictions.sql against Snowflake from the command line, so
you don't need to open the Snowflake web console. Creates GOLD_MODEL_PREDICTIONS
and GOLD_SECURITY_ALERTS (CREATE TABLE IF NOT EXISTS -- safe to re-run).

Usage:
    python warehouse/run_gold_ddl.py
"""

import os
import sys
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

DDL_FILE = Path(__file__).parent / "create_gold_predictions.sql"


def main():
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"Missing required .env values: {missing}")
        sys.exit(1)

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "NIDD_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "NIDD_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"),
    )

    sql = DDL_FILE.read_text()
    print(f"Running {DDL_FILE.name} against Snowflake ...")
    for cur in conn.execute_string(sql):
        first_line = (cur.query or "").strip().splitlines()[0][:70] if cur.query else ""
        print(f"  ok: {first_line}")
    conn.close()
    print("Gold tables ready.")


if __name__ == "__main__":
    main()
