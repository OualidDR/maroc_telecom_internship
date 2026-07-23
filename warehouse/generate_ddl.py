"""
warehouse/generate_ddl.py
===========================
Generates the Snowflake CREATE TABLE statement for the Silver table FROM
contracts/schemas.py -- same principle as Spark and Great Expectations:
one contract, every consumer derives its schema from it instead of
re-typing column lists by hand.

Usage:
    python3 warehouse/generate_ddl.py > warehouse/create_silver_flows.sql
    # then paste/run that .sql in a Snowflake worksheet (once, or whenever
    # the contract changes)
"""

import sys

sys.path.append(".")
from contracts.schemas import FEATURE_SCHEMA, LABEL_COLUMNS  # noqa: E402

TABLE_NAME = "SILVER_FLOWS"

# contract dtype -> Snowflake column type
_TYPE_MAP = {
    "float": "FLOAT",
    "int": "NUMBER(38,0)",
    "category": "VARCHAR(64)",
}


def generate_ddl() -> str:
    lines = [f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ("]

    # Event envelope metadata (added by the simulator, not in the raw CSV)
    lines.append("    event_id             VARCHAR(64) NOT NULL,")
    lines.append("    ingestion_timestamp  TIMESTAMP_NTZ NOT NULL,")
    lines.append("    schema_version       VARCHAR(16) NOT NULL,")

    # Features, straight from the contract
    for f in FEATURE_SCHEMA:
        col_type = _TYPE_MAP[f.dtype]
        nullability = "" if f.nullable else " NOT NULL"
        lines.append(f"    {f.name:20s} {col_type}{nullability},")

    # Derived indicator flags
    lines.append("    is_tcp               NUMBER(1,0) NOT NULL,")
    lines.append("    has_dst_reply        NUMBER(1,0) NOT NULL,")

    # Labels (renamed per the DELTA_INVALID_CHARACTERS fix: no spaces)
    lines.append("    Label                VARCHAR(16) NOT NULL,")
    lines.append("    attack_type          VARCHAR(32) NOT NULL,")
    lines.append("    attack_tool          VARCHAR(32) NOT NULL,")

    lines.append("    loaded_at            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()")
    lines.append(");")

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_ddl())
