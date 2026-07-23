"""
quality/expectations_silver.py
================================
Defines and runs a Great Expectations suite against the Silver Delta table
on MinIO. Most expectations are generated FROM contracts/schemas.py (not
hand-duplicated) so the suite stays in sync automatically when the contract
changes -- same principle already used in the Spark job and the simulator.

Usage:
    python3 quality/expectations_silver.py
"""

import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
from deltalake import DeltaTable
from dotenv import load_dotenv
import os

sys.path.append(".")
from contracts.schemas import FEATURE_SCHEMA, LABEL_COLUMNS  # noqa: E402
from quality.gx_context import get_context  # noqa: E402

load_dotenv()

SILVER_PATH = "s3://silver/flows"
SUITE_NAME = "silver_flows_suite"

# Domain knowledge not derivable from the contract alone -- these are the
# hand-picked rules, kept deliberately small and separate from the
# auto-generated ones below.
KNOWN_PROTO = ["icmp", "udp", "tcp", "sctp", "arp", "llc", "lldp", "ipv6-icmp"]
KNOWN_CAUSE = ["Start", "Status", "Shutdown"]
KNOWN_LABEL = ["Benign", "Malicious"]
KNOWN_ATTACK_TYPE = [
    "Benign", "UDPFlood", "HTTPFlood", "SlowrateDoS",
    "TCPConnectScan", "SYNScan", "UDPScan", "SYNFlood", "ICMPFlood",
]
KNOWN_ATTACK_TOOL = ["Benign", "Hping3", "Goldeneye", "Torshammer", "Nmap", "Slowloris"]

# Columns that must never be negative (counts, byte volumes, rates, durations).
NON_NEGATIVE_COLUMNS = [
    "Dur", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes", "DstBytes",
    "sMeanPktSz", "dMeanPktSz", "Load", "SrcLoad", "DstLoad",
    "Loss", "SrcLoss", "DstLoss", "pLoss", "Rate", "SrcRate", "DstRate",
    "TcpRtt", "SynAck", "AckDat",
]


def read_silver_sample(n: int = 50_000) -> pd.DataFrame:
    """Read the Silver Delta table from MinIO into a pandas DataFrame."""
    storage_options = {
        "AWS_ENDPOINT_URL": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        "AWS_ACCESS_KEY_ID": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "AWS_ALLOW_HTTP": "true",           # MinIO local = no TLS
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",  # required for delta-rs on MinIO
    }
    print(f"Reading Silver from {SILVER_PATH} ...")
    dt = DeltaTable(SILVER_PATH, storage_options=storage_options)
    df = dt.to_pandas()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if len(df) > n:
        df = df.sample(n, random_state=42)
        print(f"Sampled down to {n} rows for validation.")

    return df


def build_suite() -> gx.ExpectationSuite:
    """Build the expectation suite: auto-generated from the contract + hand-picked rules.

    Reused by quality/gx_spark_validator.py so the SAME suite validates both
    ad-hoc runs (this script) and every live micro-batch in the streaming job.
    """
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    expectations = []

    # --- Table-level: every column from the contract must be present -------
    all_expected_columns = (
        [f.name for f in FEATURE_SCHEMA]
        + ["is_tcp", "has_dst_reply"]
        + ["event_id", "ingestion_timestamp", "schema_version"]
        + ["Label", "attack_type", "attack_tool"]  # renamed per DELTA_INVALID_CHARACTERS fix
    )
    expectations.append(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=all_expected_columns, exact_match=False)
    )

    # --- Auto-generated from FEATURE_SCHEMA: null policy per column --------
    # Only non-nullable features get a strict not-null check. Nullable ones
    # (SrcWin, dTtl, ...) are structurally allowed to be null -- checking
    # them for not-null would be enforcing the wrong thing (see contract's
    # NULL-HANDLING POLICY).
    for f in FEATURE_SCHEMA:
        if not f.nullable:
            expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=f.name))

    # Labels and event metadata are always required.
    for col in ["Label", "attack_type", "attack_tool", "event_id", "schema_version"]:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # --- Hand-picked domain rules -------------------------------------------
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Proto", value_set=KNOWN_PROTO))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Cause", value_set=KNOWN_CAUSE))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Label", value_set=KNOWN_LABEL))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="attack_type", value_set=KNOWN_ATTACK_TYPE))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="attack_tool", value_set=KNOWN_ATTACK_TOOL))

    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="is_tcp", value_set=[0, 1]))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="has_dst_reply", value_set=[0, 1]))

    for col in NON_NEGATIVE_COLUMNS:
        expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0))

    # event_id must be unique -- catches accidental replay/duplication bugs
    # in the simulator or double-processing in Spark.
    expectations.append(gx.expectations.ExpectColumnValuesToBeUnique(column="event_id"))

    for exp in expectations:
        suite.add_expectation(exp)

    print(f"Built suite '{SUITE_NAME}' with {len(expectations)} expectations.")
    return suite


def main():
    context = get_context()
    df = read_silver_sample()

    suite = build_suite()
    context.suites.add_or_update(suite)

    data_source = context.data_sources.add_pandas(name="pandas_silver")
    asset = data_source.add_dataframe_asset(name="silver_flows_df")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="silver_flows_batch")

    validation_definition = gx.ValidationDefinition(
        name="silver_flows_validation",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add_or_update(validation_definition)

    print("\nRunning validation...")
    result = validation_definition.run(batch_parameters={"dataframe": df})

    print(f"\n{'=' * 60}")
    print(f"Overall success: {result.success}")
    print(f"{'=' * 60}")

    failed = [r for r in result.results if not r.success]
    if failed:
        print(f"\n{len(failed)} expectation(s) FAILED:\n")
        for r in failed:
            exp_type = r.expectation_config.type
            col = r.expectation_config.kwargs.get("column", "?")
            unexpected_count = r.result.get("unexpected_count", "?")
            unexpected_pct = r.result.get("unexpected_percent", "?")
            print(f"  ✗ {exp_type} on '{col}'")
            print(f"    unexpected_count={unexpected_count}  unexpected_percent={unexpected_pct}")
    else:
        print("\nAll expectations passed.")


if __name__ == "__main__":
    main()
