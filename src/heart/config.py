"""Central configuration: paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_PATH = DATA_DIR / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "target"

# Original 14 UCI attributes (the 14th, "num", is the 0-4 diagnosis).
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Valid values, used for cleaning and for API validation.
VALID_CATEGORIES = {
    "cp": [1, 2, 3, 4],        # 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic
    "restecg": [0, 1, 2],      # 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "slope": [1, 2, 3],        # 1 up, 2 flat, 3 down
    "thal": [3, 6, 7],         # 3 normal, 6 fixed defect, 7 reversible defect
}
