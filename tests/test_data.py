import numpy as np
import pandas as pd

from heart.config import FEATURES, TARGET, VALID_CATEGORIES
from heart.data import clean, mirror_to_uci


def test_target_is_binarised(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns
    # num 2 and 1 -> disease; num 0 -> no disease
    assert out.loc[0, TARGET] == 0 and out.loc[1, TARGET] == 1 and out.loc[2, TARGET] == 1


def test_duplicates_dropped(raw_df):
    assert len(clean(raw_df)) == len(raw_df) - 1


def test_impossible_and_invalid_values_become_missing(raw_df):
    out = clean(raw_df)
    assert np.isnan(out.loc[4, "chol"])          # chol == 0
    assert np.isnan(out.loc[4, "ca"])            # already missing stays missing
    assert np.isnan(out.loc[5, "thal"])          # 9 is not a valid thal code


def test_categorical_codes_valid_after_clean(raw_df):
    out = clean(raw_df)
    for col, valid in VALID_CATEGORIES.items():
        assert out[col].dropna().isin(valid).all()


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(before, raw_df)


def test_mirror_recoding_matches_uci():
    mirror = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 4, "thal": 1, "target": 1,
    }])
    row = mirror_to_uci(mirror).iloc[0]
    assert row["cp"] == 1 and row["restecg"] == 2 and row["slope"] == 3
    assert row["thal"] == 6 and np.isnan(row["ca"]) and row["num"] == 0


def test_cleaned_dataset_schema(clean_df):
    assert len(clean_df) > 290
    assert set(FEATURES + [TARGET]) == set(clean_df.columns)
    assert clean_df[TARGET].between(0, 1).all()
    assert clean_df.isna().sum().sum() < 10
