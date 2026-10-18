import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heart.config import FEATURES
from heart.features import build_pipeline, build_preprocessor, split_xy


def test_preprocessor_output_has_no_nans(clean_df):
    X, _ = split_xy(clean_df)
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    # 6 numeric + 3 binary + one-hot(cp 4 + restecg 3 + slope 3 + thal 3)
    assert Xt.shape == (len(X), 6 + 3 + 13)


def test_numeric_features_are_scaled(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    num = Xt[:, :6]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-2)


def test_handles_missing_and_unseen_values(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    odd = X.head(2).copy()
    odd.loc[odd.index[0], ["ca", "thal"]] = np.nan
    odd.loc[odd.index[1], "cp"] = 99          # unseen category is ignored, not an error
    proba = pipe.predict_proba(odd)[:, 1]
    assert proba.shape == (2,) and np.all((proba >= 0) & (proba <= 1))


def test_column_order_does_not_matter(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    shuffled = X[list(reversed(FEATURES))]
    np.testing.assert_allclose(pipe.predict_proba(X), pipe.predict_proba(shuffled))


def test_split_xy(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURES and isinstance(y, pd.Series)
