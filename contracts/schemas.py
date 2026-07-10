"""
contracts/schema.py
====================
Data contract for the 5G-NIDD dataset (Combined.csv).

This is the SHARED source of truth between the Data Engineering pipeline
(ingestion, Spark, Kafka schema) and the Data Science pipeline (feature
engineering, training, inference). Both sides import from here instead of
hardcoding column names/types.

Built from real exploration of Combined.csv (1,215,890 rows x 52 raw columns).
See explor_full.py for the full analysis that justified each decision below.
"""

from dataclasses import dataclass
from typing import Literal


# -----------------------------------------------------------------------------
# 1. RAW COLUMNS DROPPED BEFORE ANYTHING ELSE (never enter the pipeline)
# -----------------------------------------------------------------------------
# Reason for each drop is documented -- do not remove this list, it's part of
# the audit trail for the final report.

DROPPED_COLUMNS = {
    "Unnamed: 0": "Leftover pandas index from a previous export, not a real feature.",
    "Seq": (
        "Not a usable chronological identifier. Resets ~128,379 times across "
        "1.2M rows (Argus internal batch bookkeeping), unrelated to attack "
        "sessions or real-world ordering."
    ),
    "RunTime": "100% identical to Dur on every row -- redundant.",
    "Mean": "100% identical to Dur on every row -- redundant (Argus aggregate field, unused for single-flow rows).",
    "Sum": "100% identical to Dur on every row -- redundant.",
    "Min": "100% identical to Dur on every row -- redundant.",
    "Max": "100% identical to Dur on every row -- redundant.",
    "sVid": "Zero variance (constant = 610 where not null) and >90% missing -- no signal.",
    "dVid": "Zero variance (constant = 610 where not null) and >99% missing -- no signal.",
}


# -----------------------------------------------------------------------------
# 2. LABEL COLUMNS (targets, never used as input features)
# -----------------------------------------------------------------------------

LABEL_COLUMNS = {
    "Label": "Binary target: 'Benign' or 'Malicious'.",
    "Attack Type": (
        "Multiclass target (9 classes): Benign, UDPFlood, HTTPFlood, "
        "SlowrateDoS, TCPConnectScan, SYNScan, UDPScan, SYNFlood, ICMPFlood."
    ),
    "Attack Tool": (
        "Metadata only -- which tool generated the flow (Hping3, Goldeneye, "
        "Nmap, Slowloris, Torshammer, or Benign). Strongly label-correlated: "
        "exclude from training features (same leakage logic as Attack Type)."
    ),
}


# -----------------------------------------------------------------------------
# 3. FEATURE COLUMNS (the 39 columns allowed as model input)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: Literal["float", "int", "category"]
    nullable: bool
    notes: str = ""


FEATURE_SCHEMA: list[FeatureSpec] = [
    # --- duration ---
    FeatureSpec("Dur", "float", False, "Flow duration in seconds."),

    # --- protocol / connection state ---
    FeatureSpec("Proto", "category", False,
                "8 values: icmp, udp, tcp, sctp, arp, llc, lldp, ipv6-icmp."),
    FeatureSpec("Cause", "category", False,
                "3 values: Start, Status, Shutdown (Argus record cause)."),
    FeatureSpec("State", "category", False,
                "11 values, e.g. ECO, CON, REQ, FIN, RST -- Argus flow state."),

    # --- type of service ---
    FeatureSpec("sTos", "float", True, "Src Type of Service. ~214 nulls only."),
    FeatureSpec("dTos", "float", True, "Dst Type of Service. Null when no dst reply (~77%)."),
    FeatureSpec("sDSb", "category", True, "Src DSCP class (12 values). ~214 nulls."),
    FeatureSpec("dDSb", "category", True, "Dst DSCP class (6 values). Null when no dst reply (~77%)."),

    # --- TTL / hops ---
    FeatureSpec("sTtl", "float", True, "Src TTL. ~214 nulls only."),
    FeatureSpec("dTtl", "float", True, "Dst TTL. Null when no dst reply (~77%)."),
    FeatureSpec("sHops", "float", True, "Src hop count estimate. ~214 nulls only."),
    FeatureSpec("dHops", "float", True, "Dst hop count estimate. Null when no dst reply (~77%)."),

    # --- packet / byte counts ---
    FeatureSpec("TotPkts", "int", False, "Total packets, both directions."),
    FeatureSpec("SrcPkts", "int", False, "Packets sent by source."),
    FeatureSpec("DstPkts", "int", False, "Packets sent by destination."),
    FeatureSpec("TotBytes", "int", False, "Total bytes, both directions."),
    FeatureSpec("SrcBytes", "int", False, "Bytes sent by source."),
    FeatureSpec("DstBytes", "int", False, "Bytes sent by destination."),
    FeatureSpec("Offset", "int", False, "Argus internal byte offset."),
    FeatureSpec("sMeanPktSz", "float", False, "Mean packet size, source side."),
    FeatureSpec("dMeanPktSz", "float", False, "Mean packet size, destination side."),

    # --- load / rate / loss ---
    FeatureSpec("Load", "float", False, "Total bits/sec throughput."),
    FeatureSpec("SrcLoad", "float", False, "Src-side bits/sec."),
    FeatureSpec("DstLoad", "float", False, "Dst-side bits/sec."),
    FeatureSpec("Loss", "int", False, "Total packets lost/retransmitted."),
    FeatureSpec("SrcLoss", "int", False, "Src-side loss."),
    FeatureSpec("DstLoss", "int", False, "Dst-side loss."),
    FeatureSpec("pLoss", "float", False, "Percentage loss."),
    FeatureSpec("SrcGap", "float", True, "Src-side packet gap (TCP only, ~77% null)."),
    FeatureSpec("DstGap", "float", True, "Dst-side packet gap (TCP only, ~77% null)."),
    FeatureSpec("Rate", "float", False, "Total packets/sec."),
    FeatureSpec("SrcRate", "float", False, "Src-side packets/sec."),
    FeatureSpec("DstRate", "float", False, "Dst-side packets/sec."),

    # --- TCP-specific (mostly null for udp/icmp -- expected, not a data bug) ---
    FeatureSpec("SrcWin", "float", True, "TCP window size, source. Null for non-TCP (~87-100%)."),
    FeatureSpec("DstWin", "float", True, "TCP window size, destination. Null for non-TCP."),
    FeatureSpec("SrcTCPBase", "float", True,
                "TCP initial sequence number, source. Range up to 2^32-1 (legit uint32, not a sentinel)."),
    FeatureSpec("DstTCPBase", "float", True, "TCP initial sequence number, destination."),
    FeatureSpec("TcpRtt", "float", False, "TCP round-trip time (0.0 for non-TCP)."),
    FeatureSpec("SynAck", "float", False, "SYN-ACK handshake timing (0.0 for non-TCP)."),
    FeatureSpec("AckDat", "float", False, "ACK-data handshake timing (0.0 for non-TCP)."),
]

FEATURE_COLUMNS = [f.name for f in FEATURE_SCHEMA]


# -----------------------------------------------------------------------------
# 4. NULL-HANDLING POLICY
# -----------------------------------------------------------------------------
# NaNs in this dataset are STRUCTURAL, not missing-at-random:
#   - d* columns (dTos, dTtl, dHops, dDSb) are null when the destination never
#     replied (e.g. one-way UDP/ICMP traffic).
#   - TCP-only columns (SrcWin, DstWin, SrcTCPBase, DstTCPBase, SrcGap, DstGap)
#     are null for udp/icmp/arp/... flows by definition.
#
# Policy: DO NOT impute with mean/median. Use a sentinel value + an explicit
# boolean indicator column instead, so the model can distinguish
# "not applicable" from "applicable but zero".

DERIVED_INDICATOR_COLUMNS = {
    "is_tcp": "1 if Proto == 'tcp' else 0. Explains nulls in TCP-only columns.",
    "has_dst_reply": "1 if dTtl is not null else 0. Explains nulls in d* columns.",
}

NULL_SENTINEL = -1.0  # applied to numeric feature columns only, after adding indicators


# -----------------------------------------------------------------------------
# 5. UNIFIED KAFKA EVENT WRAPPER (matches the brief's event format)
# -----------------------------------------------------------------------------

EVENT_WRAPPER_EXAMPLE = {
    "event_id": "evt_00000123",
    "ingestion_timestamp": "2026-07-09T12:00:00.000Z",
    "schema_version": "1.0.0",
    "flow": "{...39 raw feature fields + 3 label fields...}",
}

SCHEMA_VERSION = "1.0.0"


if __name__ == "__main__":
    print(f"Feature columns ({len(FEATURE_COLUMNS)}):")
    for f in FEATURE_SCHEMA:
        print(f"  - {f.name:15s} {f.dtype:9s} nullable={f.nullable}")
    print(f"\nLabel columns ({len(LABEL_COLUMNS)}): {list(LABEL_COLUMNS)}")
    print(f"\nDropped columns ({len(DROPPED_COLUMNS)}): {list(DROPPED_COLUMNS)}")
