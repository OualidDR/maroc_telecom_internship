import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import DEFAULT_CSV_PATH, load_or_build

df = load_or_build(DEFAULT_CSV_PATH)
print(f"Shape after preprocessing: {df.shape}")
print(df["Attack Type"].value_counts())

