"""Download the dataset from Kaggle into data/raw/.

Dataset: "Sri Lanka Weather Dataset" by rasulmah (CC0 public domain).
A copy of the CSV is already in data/raw/, so this script is only needed
if you want to download it again. It needs a Kaggle login (kagglehub).
"""
import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("rasulmah/sri-lanka-weather-dataset")
print("Path to dataset files:", path)

raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
for f in Path(path).rglob("*"):
    if f.is_file():
        shutil.copy2(f, raw_dir / f.name)
        print("Copied:", f.name)
