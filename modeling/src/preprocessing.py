import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from contracts.schemas import (
    DROPPED_COLUMNS,
    FEATURE_SCHEMA,
    NULL_SENTINEL,
)

def load_clean_csv(csv_path: str) -> pd.DataFrame:
    
    df=pd.read_csv(csv_path, low_memory=False)
    
    # Drop junk columns
    df = df.drop(columns=[col for col in DROPPED_COLUMNS if col in df.columns])
    
    # Add structural indicators
    df["is_tcp"] = (df["Proto"] == "tcp").astype(int)
    df["has_dst_reply"] = df["dTtl"].notna().astype(int)
    
    # Fill NaNs in numeric feature columns with sentinel value
    numeric_cols = [f.name for f in FEATURE_SCHEMA if f.dtype in ("float", "int")]
    df[numeric_cols] = df[numeric_cols].fillna(NULL_SENTINEL)
    
    # One-hot encode categorical columns
    categorical_cols = [f.name for f in FEATURE_SCHEMA if f.dtype == "category"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    return df

if __name__ == "__main__":
    df = load_clean_csv("../../data/raw/Combined.csv")
    print(f"Shape: {df.shape}")
    print(f"Any nulls left in numeric features: "
          f"{df.select_dtypes('number').isnull().sum().sum()}")
    


