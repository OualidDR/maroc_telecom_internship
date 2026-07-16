"""
Engineered features targeting the Benign vs UDPFlood confusion.

Rationale:
The three baseline models (LogReg, RF, XGBoost) all converge on the same
trade-off at this class boundary, suggesting the confusion is fundamental
to the feature representation rather than a model expressiveness limit.
These engineered features are designed to capture behavioral signals that
distinguish flood traffic (uniform, one-directional, high-rate) from heavy
legitimate UDP traffic (variable, bidirectional).

Added features:
- bytes_per_packet:      Total bytes / total packets. Floods often use
                         uniform small packets; legitimate traffic varies.
- src_dst_pkt_ratio:     SrcPkts / DstPkts. Floods are near-one-way (attacker
                         pumps, destination silent); legitimate flows balance.
- src_dst_byte_ratio:    SrcBytes / DstBytes. Same directionality signal on
                         a volume basis.
- log_rate:              log1p(Rate). Rate spans 5+ orders of magnitude;
                         log-scale exposes low-value patterns to the model.
- log_totbytes:          log1p(TotBytes). Same reasoning for byte volume.
- log_totpkts:           log1p(TotPkts). Same reasoning for packet volume.

All ratios use +1 denominator smoothing to avoid division by zero on flows
where one side sent nothing.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features that target the Benign vs UDPFlood boundary.

    Called after load_and_clean() so all sentinel fills and indicators are
    already in place. Sentinel values (-1) will propagate through ratios;
    that's acceptable because the is_tcp / has_dst_reply indicators still
    tell the model which rows those sentinels came from.
    """
    df = df.copy()

    # Ratios (directionality and packet-size uniformity)
    df["bytes_per_packet"] = df["TotBytes"] / (df["TotPkts"] + 1)
    df["src_dst_pkt_ratio"] = df["SrcPkts"] / (df["DstPkts"] + 1)
    df["src_dst_byte_ratio"] = df["SrcBytes"] / (df["DstBytes"] + 1)

    # Log-scaled volume features (compress skewed distributions)
    df["log_rate"] = np.log1p(df["Rate"].clip(lower=0))
    df["log_totbytes"] = np.log1p(df["TotBytes"].clip(lower=0))
    df["log_totpkts"] = np.log1p(df["TotPkts"].clip(lower=0))

    return df