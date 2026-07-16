import sys
from pathlib import Path
from .feature_engineering import add_engineered_features

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

import pandas as pd
from contracts.schemas import (
    DROPPED_COLUMNS,
    FEATURE_SCHEMA,
    NULL_SENTINEL,
)


def load_clean_csv(csv_path: str, engineer: bool = True) -> pd.DataFrame:
    """Load Combined.csv and apply the contract preprocessing.

    Args:
        csv_path: path to Combined.csv
        engineer: if True, add engineered features from feature_engineering.py
    """
    df = pd.read_csv(csv_path, low_memory=False)

    df = df.drop(columns=list(DROPPED_COLUMNS.keys()))

    df["is_tcp"] = (df["Proto"] == "tcp").astype(int)
    df["has_dst_reply"] = df["dTtl"].notna().astype(int)

    numeric_cols = [f.name for f in FEATURE_SCHEMA if f.dtype in ("float", "int")]
    df[numeric_cols] = df[numeric_cols].fillna(NULL_SENTINEL)

    categorical_cols = [f.name for f in FEATURE_SCHEMA if f.dtype == "category"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    if engineer:
        df = add_engineered_features(df)

    return df


DEFAULT_CACHE_PATH = REPO_ROOT / "modeling" / "artifacts" / "processed" / "clean.parquet"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "raw" / "Combined.csv"


def load_or_build(
    csv_path: str,
    cache_path: str = None,
    engineer: bool = True,
) -> pd.DataFrame:
    """Load preprocessed data from cache if available, else build and cache."""
    if cache_path is None:
        cache_path = (
            "modeling/artifacts/processed/clean_engineered.parquet"
            if engineer
            else "modeling/artifacts/processed/clean.parquet"
        )

    cache = Path(cache_path)

    if cache.exists():
        print(f"Loading cached preprocessed data from {cache}")
        return pd.read_parquet(cache)

    print(f"No cache found at {cache} — running preprocessing...")
    df = load_clean_csv(csv_path, engineer=engineer)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"Cached to {cache}")
    return df


if __name__ == "__main__":
    df = load_or_build(DEFAULT_CSV_PATH)
    print(f"Shape: {df.shape}")
    print(
        f"Any nulls left in numeric features: "
        f"{df.select_dtypes('number').isnull().sum().sum()}"
    )
