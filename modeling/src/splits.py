from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLS = ["Label", "Attack Type", "Attack Tool"]

RANDOM_STATE = 42

def make_splits(df:pd.DataFrame, target='Attack Type'):
    """Return X_train, X_val, X_test, y_train, y_val, y_test.

    70/15/15 stratified split. Uses RANDOM_STATE = 42 for reproducibility.
    """
    
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols]
    y = df[target]

    # First split: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # Second split: halve the 30% into 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(splits: tuple, out_dir: str = "modeling/artifacts/splits"):
    """Save all six arrays as parquet files for fast reload."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = splits
    X_train.to_parquet(out / "X_train.parquet")
    X_val.to_parquet(out / "X_val.parquet")
    X_test.to_parquet(out / "X_test.parquet")
    y_train.to_frame().to_parquet(out / "y_train.parquet")
    y_val.to_frame().to_parquet(out / "y_val.parquet")
    y_test.to_frame().to_parquet(out / "y_test.parquet")

    print(f"Splits saved to {out}/")
    
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from modeling.src.preprocessing import load_or_build

    df = load_or_build("data/raw/Combined.csv")
    splits = make_splits(df)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    print("\nTest set class distribution (verify stratification):")
    print(y_test.value_counts())

    save_splits(splits)
    
    
    
    