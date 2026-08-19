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
import math
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

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


def row_to_event(flow: dict, event_index: int, ingestion_ts: str | None = None) -> dict:
    """Wrap one row (already a plain dict) into the shared Kafka event envelope.

    Expects `flow` from df.to_dict('records') -- NaN values still need
    converting to None (JSON has no NaN), and numpy scalar types still need
    normalizing to native Python types for the JSON serializer.

    ingestion_ts: if given, used as the event's ingestion_timestamp instead of
    "now" -- lets the caller spread events across a synthetic time window (see
    --spread-hours) so downstream hourly dashboards aren't all one bucket.
    """
    flow = {
        k: (None if isinstance(v, float) and pd.isna(v) else (v.item() if hasattr(v, "item") else v))
        for k, v in flow.items()
    }

    return {
        "event_id": f"evt_{event_index:08d}_{uuid.uuid4().hex[:8]}",
        "ingestion_timestamp": ingestion_ts or datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "flow": flow,
    }


# Reconnaissance classes get their own early "scan campaign" spike; everything
# else malicious (floods / DoS) clusters in later "attack wave" spikes.
_RECON_CLASSES = {"SYNScan", "UDPScan", "TCPConnectScan"}


def build_synthetic_timestamp_fn(spread_hours: float, base: datetime, seed: int):
    """Return fn(attack_type) -> ISO timestamp string placing each event on a
    SYNTHETIC but realistic-looking SOC timeline.

    IMPORTANT: these timestamps are fabricated for the demo dashboard. The
    source CSV has no recoverable chronological order (see the exploration
    report), so there is no real capture time to preserve. This deliberately
    shapes a narrative instead of a flat line:

      - Benign traffic follows a diurnal day/night wave (busy ~15:00, quiet ~03:00).
      - Reconnaissance scans cluster in one early "scan campaign" spike.
      - Floods / DoS cluster in three later "attack wave" spikes of varying size.

    The result on the Power BI SOC pages: a calm daily baseline punctuated by a
    recon probe and then DDoS bursts -- an attack story, not noise. Documented
    as synthetic in the report.
    """
    span = spread_hours * 3600.0
    n_hours = max(1, int(math.ceil(spread_hours)))
    t0 = base - timedelta(seconds=span)
    rng = random.Random(seed)

    # Diurnal benign weight per hour bucket (cosine peaking at 15:00, floored so
    # night never fully drops to zero).
    hours = list(range(n_hours))
    benign_weights = [
        max(0.05, 0.55 + 0.45 * math.cos(2 * math.pi * ((t0 + timedelta(hours=h)).hour - 15) / 24))
        for h in hours
    ]

    # Attack campaign windows as (center_fraction, sigma_fraction) of the span.
    recon_window = (0.18, 0.03)
    ddos_windows = [(0.36, 0.028), (0.58, 0.022), (0.82, 0.030)]
    ddos_weights = [0.40, 0.35, 0.25]

    def _gauss_offset(center: float, sigma: float) -> float:
        frac = min(1.0, max(0.0, rng.gauss(center, sigma)))
        return frac * span

    def _timestamp_for(attack_type) -> str:
        if attack_type is None or attack_type == "Benign":
            hb = rng.choices(hours, weights=benign_weights, k=1)[0]
            offset = min(hb * 3600 + rng.uniform(0, 3600), span)
        elif attack_type in _RECON_CLASSES:
            offset = _gauss_offset(*recon_window)
        else:  # floods / DoS
            center, sigma = rng.choices(ddos_windows, weights=ddos_weights, k=1)[0]
            offset = _gauss_offset(center, sigma)
        return (t0 + timedelta(seconds=offset)).isoformat()

    return _timestamp_for


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
    parser.add_argument("--spread-hours", type=float, default=0.0,
                         help="Spread ingestion_timestamps randomly across the last N hours "
                              "instead of stamping them all 'now'. Use for demo dashboards so "
                              "hourly time-series aren't collapsed into a single bucket "
                              "(e.g. --spread-hours 48). 0 = real now (default).")
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

    # .to_dict('records') once, upfront -- roughly 50-60x faster than
    # iterrows() per-row conversion (measured), which matters a lot when
    # streaming the full 1.2M-row dataset instead of a small test slice.
    records = df.to_dict("records")

    # Synthetic ingestion time window. When --spread-hours > 0, each event is
    # placed on a fabricated but realistic SOC timeline (diurnal benign baseline
    # + recon/DDoS attack spikes -- see build_synthetic_timestamp_fn), keyed on
    # the event's Attack Type. Seeded => reproducible. Documented as synthetic
    # (the source CSV has no real chronological order -- see exploration report).
    if args.spread_hours > 0:
        ts_fn = build_synthetic_timestamp_fn(args.spread_hours, datetime.now(timezone.utc), args.seed)
        print(f"Synthetic narrative timeline over the last {args.spread_hours:g}h "
              f"(diurnal benign baseline + recon scan spike + DDoS attack waves).")
    else:
        ts_fn = None

    try:
        for i, flow in enumerate(records):
            if _shutdown_requested:
                break

            ingestion_ts = ts_fn(flow.get("Attack Type")) if ts_fn else None

            event = row_to_event(flow, i, ingestion_ts)
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