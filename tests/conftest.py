import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart.config import CLEAN_PATH, COLUMNS
from heart.features import build_pipeline, split_xy

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small raw frame in the original UCI coding, including messy values."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 0],
        [41, 0, 2, 130, 0, 0, 2, 172, 0, 1.4, 1, np.nan, 3, 0],   # chol 0, ca missing
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 9, 4],      # invalid thal
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 9, 4],      # duplicate
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    if CLEAN_PATH.exists():
        return pd.read_csv(CLEAN_PATH)
    pytest.skip("cleaned dataset not present; run `python -m heart.data` first")


@pytest.fixture(scope="session")
def tiny_model_files(tmp_path_factory, clean_df):
    """Fit a fast model and write it where the API can load it."""
    import joblib

    d = tmp_path_factory.mktemp("model")
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    joblib.dump(pipe, d / "model.joblib")
    (d / "model_meta.json").write_text(json.dumps(
        {"model_name": "test_lr", "mlflow_run_id": "testrun123"}))
    return d / "model.joblib", d / "model_meta.json"
