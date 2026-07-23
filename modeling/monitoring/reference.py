import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SPLITS_DIR = REPO_ROOT / "modeling" / "artifacts" / "splits"
REFERENCE_PATH = REPO_ROOT / "modeling" / "artifacts" / "monitoring" / "reference.parquet"


def build_reference():
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet").squeeze()

    # For monitoring, we want the reference frame to include the *label*
    # column so Evidently knows the "expected" prediction distribution.
    reference = X_train.copy()
    reference["prediction"] = y_train.values

    # Sample down to a manageable size — full 850K rows is overkill and slow.
    # Stratified sample preserves class balance.
    sample = reference.groupby("prediction", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 5000), random_state=42)
    )

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(REFERENCE_PATH)

    print(f"Reference dataset saved to {REFERENCE_PATH}")
    print(f"Shape: {sample.shape}")
    print(f"Class distribution:\n{sample['prediction'].value_counts()}")


if __name__ == "__main__":
    build_reference()