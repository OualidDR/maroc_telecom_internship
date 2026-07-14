import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_clean_csv

df = load_clean_csv("../../data/raw/Combined.csv")
print(f"Shape after preprocessing: {df.shape}")
print(df["Attack Type"].value_counts())

