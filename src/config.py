"""Central settings: every script and the app read paths and constants from here."""
from pathlib import Path

RANDOM_STATE = 42          # one seed everywhere so results can be reproduced
N_JOBS = -1                # use all CPU cores when training

# ---- folders -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---- files ---------------------------------------------------------------
RAW_FILE = RAW_DIR / "SriLanka_Weather_Dataset.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
INFERENCE_SAMPLE_FILE = PROCESSED_DIR / "inference_sample.csv"
MODEL_FILE = MODELS_DIR / "model.pkl"              # soft-voting ensemble (used by the app)
FEATURE_INFO_FILE = MODELS_DIR / "feature_info.json"

# ---- problem definition --------------------------------------------------
# A "rainy day" is a day with at least 1.0 mm of rain. This is the standard
# wet-day threshold used in climate studies (WMO / ETCCDI index "R1mm").
# Smaller amounts are drizzle that the weather model often reports as noise.
RAIN_THRESHOLD_MM = 1.0
TARGET = "rain_tomorrow"
CLASS_LABELS = {0: "No rain", 1: "Rain"}

# The latest 20% of dates are kept for testing (chronological split).
TEST_FRACTION = 0.20

# The single record kept out of training AND testing for the inference demo.
HOLDOUT_CITY = "Colombo"
