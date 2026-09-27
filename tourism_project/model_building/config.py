"""Shared configuration for every stage of the MLOps pipeline.

All paths are resolved relative to this file, so the scripts behave the same
from a notebook, a terminal, or a GitHub Actions runner.
"""
from pathlib import Path

# ------------------------------------------------------------------ paths
PROJECT_DIR = Path(__file__).resolve().parents[1]       # tourism_project/
REPO_DIR = PROJECT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"         # registered dataset
SPLIT_DIR = PROJECT_DIR / "data" / "processed"           # train/test splits (artifact)
DEPLOY_DIR = PROJECT_DIR / "deployment"                  # app + committed model
MODEL_PATH = DEPLOY_DIR / "model.joblib"
METRICS_PATH = DEPLOY_DIR / "metrics.json"
MLFLOW_URI = f"sqlite:///{REPO_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "wellness-tourism-prodtaken"

# ------------------------------------------------------------------ schema
TARGET = "ProdTaken"

# Expected type of every column in the business data dictionary.
# Validation fails if a column is missing OR has changed type.
EXPECTED_DTYPES = {
    "CustomerID": "numeric", "ProdTaken": "numeric", "Age": "numeric",
    "TypeofContact": "categorical", "CityTier": "numeric",
    "DurationOfPitch": "numeric", "Occupation": "categorical",
    "Gender": "categorical", "NumberOfPersonVisiting": "numeric",
    "NumberOfFollowups": "numeric", "ProductPitched": "categorical",
    "PreferredPropertyStar": "numeric", "MaritalStatus": "categorical",
    "NumberOfTrips": "numeric", "Passport": "numeric",
    "PitchSatisfactionScore": "numeric", "OwnCar": "numeric",
    "NumberOfChildrenVisiting": "numeric", "Designation": "categorical",
    "MonthlyIncome": "numeric",
}
EXPECTED_COLUMNS = list(EXPECTED_DTYPES)

# Columns removed during cleaning:
#   CustomerID     -> unique identifier, no predictive value
#   ProductPitched -> maps one-to-one to Designation (fully redundant)
DROP_COLUMNS = ["CustomerID", "ProductPitched"]

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation",
]

# Known only AFTER a sales pitch. Set EXCLUDE_POST_CONTACT=true to train a
# model that can score brand-new leads before anyone contacts them.
POST_CONTACT_FEATURES = [
    "DurationOfPitch", "NumberOfFollowups", "PitchSatisfactionScore",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
