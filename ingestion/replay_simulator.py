"""
ingestion/replay_simulator.py
==============================
Reads the 5G-NIDD raw CSV, applies the data contract (contracts/schemas.py),
and streams each row as a JSON event to a Kafka/Redpanda topic at a
controllable rate.

Decision (see explor_full.py / rapport_contrat_donnees_5G-NIDD.md):
No real chronological order is recoverable from this file (Seq resets
~128k times, Attack Type blocks are scattered). So this simulator SHUFFLES
rows rather than replaying them in raw file order, while preserving the
real class proportions -- and documents this as a data limitation, not
a pipeline bug.

Usage:
    python3 ingestion/replay_simulator.py --rate 100
    python3 ingestion/replay_simulator.py --rate 500 --limit 5000   # quick test
    python3 ingestion/replay_simulator.py --rate 50 --topic flows-raw --bootstrap-servers localhost:9092
"""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()  # reads .env if present, falls back to shell env / hardcoded defaults below

# Make sure this script can find contracts/ whether run from repo root or
# from ingestion/. Adjust if your repo layout differs.
sys.path.append(".")
from contracts.schemas import (  # noqa: E402
    DROPPED_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    SCHEMA_VERSION,
)

DATA_PATH = "data/raw/Combined.csv"
DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Graceful shutdown flag, flipped by SIGINT/SIGTERM handler below.
_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    print("\nShutdown requested, finishing current batch and flushing...")
    _shutdown_requested = True


def load_and_prepare(limit: int | None, seed: int) -> pd.DataFrame:
    """Load the CSV, apply the data contract, shuffle, and add derived flags."""
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Apply the contract: drop the 9 columns we decided are not features.
    to_drop = [c for c in DROPPED_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)

    # Derived indicator columns (documented in schemas.py NULL-HANDLING POLICY).
    df["is_tcp"] = (df["Proto"] == "tcp").astype(int)
    df["has_dst_reply"] = df["dTtl"].notna().astype(int)

    # Shuffle -- row order in the raw file carries no real chronological
    # signal (see exploration report), so we don't pretend otherwise.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if limit:
        df = df.head(limit)
        print(f"Limiting to first {limit} shuffled rows (test mode).")

    print(f"Ready to stream {len(df)} rows "
          f"({len(FEATURE_COLUMNS)} features + {len(LABEL_COLUMNS)} labels + 2 derived flags).")
    return df


def row_to_event(row: pd.Series, event_index: int) -> dict:
    """Wrap one row into the shared Kafka event envelope."""
    flow = row.where(pd.notnull(row), None).to_dict()
    # numpy scalar types aren't JSON-serializable as-is; normalize them.
    flow = {k: (v.item() if hasattr(v, "item") else v) for k, v in flow.items()}

    return {
        "event_id": f"evt_{event_index:08d}_{uuid.uuid4().hex[:8]}",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "flow": flow,
    }


def main():
    parser = argparse.ArgumentParser(description="5G-NIDD replay simulator (CSV -> Kafka/Redpanda)")
    parser.add_argument("--rate", type=float, default=100,
                         help="Target events per second (default: 100)")
    parser.add_argument("--topic", type=str, default="flows-raw",
                         help="Kafka topic to publish to (default: flows-raw)")
    parser.add_argument("--bootstrap-servers", type=str, default=DEFAULT_BOOTSTRAP,
                         help=f"Kafka/Redpanda bootstrap servers (default: {DEFAULT_BOOTSTRAP}, from .env)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only stream the first N shuffled rows (useful for a quick test)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for the shuffle (default: 42)")
    parser.add_argument("--log-every", type=int, default=1000,
                         help="Print progress every N events (default: 1000)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    df = load_and_prepare(limit=args.limit, seed=args.seed)

    print(f"Connecting to Kafka/Redpanda at {args.bootstrap_servers} ...")
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        linger_ms=20,       # small batching window, cheap on a low-core machine
        retries=3,
    )

    sleep_interval = 1.0 / args.rate if args.rate > 0 else 0
    sent = 0
    start_time = time.time()

    try:
        for i, row in df.iterrows():
            if _shutdown_requested:
                break

            event = row_to_event(row, i)
            producer.send(args.topic, key=event["event_id"], value=event)
            sent += 1

            if sent % args.log_every == 0:
                elapsed = time.time() - start_time
                actual_rate = sent / elapsed if elapsed > 0 else 0
                print(f"  sent={sent}  elapsed={elapsed:.1f}s  actual_rate={actual_rate:.1f} evt/s")

            if sleep_interval:
                time.sleep(sleep_interval)

    finally:
        print("Flushing producer...")
        producer.flush(timeout=30)
        producer.close()
        elapsed = time.time() - start_time
        print(f"Done. Sent {sent} events in {elapsed:.1f}s "
              f"(avg {sent / elapsed if elapsed else 0:.1f} evt/s) to topic '{args.topic}'.")


if __name__ == "__main__":
    main()