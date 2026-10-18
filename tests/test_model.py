import joblib
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from heart.config import MODEL_PATH, RANDOM_STATE
from heart.evaluate import compute_metrics
from heart.features import build_pipeline, split_xy
from heart.train import feature_importance, model_space


def test_model_space_contains_required_models():
    space = model_space()
    assert {"logistic_regression", "random_forest"} <= set(space)
    assert len(space) >= 2


def test_quick_grids_have_one_candidate():
    for _, grid in model_space(quick=True).values():
        assert all(len(v) == 1 for v in grid.values())


@pytest.mark.parametrize("name", list(model_space(quick=True)))
def test_each_family_trains_and_beats_chance(name, clean_df):
    est, grid = model_space(quick=True)[name]
    X, y = split_xy(clean_df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=RANDOM_STATE)
    pipe = build_pipeline(est).set_params(**{k: v[0] for k, v in grid.items()})
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    assert roc_auc_score(y_te, proba) > 0.75


def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert all(v == 1.0 for v in m.values())


def test_feature_importance_for_linear_model(clean_df):
    est, grid = model_space(quick=True)["logistic_regression"]
    X, y = split_xy(clean_df)
    pipe = build_pipeline(est).set_params(**{k: v[0] for k, v in grid.items()}).fit(X, y)
    names, values = feature_importance(pipe)
    assert len(names) == len(values) > 0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model yet")
def test_saved_model_quality(clean_df):
    pipe = joblib.load(MODEL_PATH)
    X, y = split_xy(clean_df)
    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) > 0.85


def test_one_se_rule_prefers_simpler_model_within_noise():
    from heart.train import select_champion

    results = [
        {"name": "random_forest", "cv_roc_auc": 0.92, "cv_roc_auc_std": 0.03},
        {"name": "logistic_regression", "cv_roc_auc": 0.90, "cv_roc_auc_std": 0.03},
        {"name": "knn", "cv_roc_auc": 0.85, "cv_roc_auc_std": 0.03},
    ]
    assert select_champion(results)["name"] == "logistic_regression"
    results[1]["cv_roc_auc"] = 0.80          # now clearly worse -> RF wins
    assert select_champion(results)["name"] == "random_forest"
