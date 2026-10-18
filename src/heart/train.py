"""Train, tune and compare classifiers with full MLflow tracking.

For every model family:
  1. GridSearchCV (5-fold stratified CV, refit on ROC-AUC) on the training split
  2. every grid candidate is logged as a nested MLflow child run
  3. the best candidate is evaluated on the held-out test split and logged
     with params, CV + test metrics, plots and the fitted sklearn pipeline
The family with the best mean CV ROC-AUC is saved as the production model.

Usage:  python -m heart.train [--quick]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from heart import evaluate
from heart.config import (
    CLEAN_PATH, FEATURES, FIG_DIR, MODEL_DIR, MODEL_META_PATH, MODEL_PATH, RANDOM_STATE,
)
from heart.data import load_clean
from heart.features import build_pipeline, split_xy

log = logging.getLogger(__name__)
EXPERIMENT = "heart-disease-classification"


def model_space(quick: bool = False) -> dict[str, tuple[object, dict]]:
    """Estimator + hyper-parameter grid per model family."""
    space = {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE),
            # l1_ratio 0 = ridge (L2), 1 = lasso (L1), 0.5 = elastic net
            {"model__C": [1.0, 0.5, 0.1, 0.01, 5.0], "model__l1_ratio": [0.0, 0.5, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3, 5]},
        ),
        "xgboost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=2),
            {"model__n_estimators": [100, 300], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0]},
        ),
        "svm_rbf": (
            # SVC has no native probabilities; calibrate (Platt sigmoid) to get confidence
            CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False, cv=5),
            {"model__estimator__C": [1.0, 0.1, 0.5, 5.0],
             "model__estimator__gamma": ["scale", 0.01, 0.05]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 9, 15, 21], "model__weights": ["uniform", "distance"]},
        ),
    }
    if quick:  # tiny grids for CI smoke runs
        space = {k: (est, {p: v[:1] for p, v in grid.items()}) for k, (est, grid) in space.items()}
    return space


def feature_importance(pipe) -> tuple[list[str], np.ndarray] | None:
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return list(names), model.feature_importances_
    if hasattr(model, "coef_"):
        return list(names), model.coef_.ravel()
    return None


def _clean_params(params: dict) -> dict:
    """'model__estimator__C' -> 'C' for readable MLflow params."""
    return {k.replace("model__", "").replace("estimator__", ""): v for k, v in params.items()}


def _log_candidates(search: GridSearchCV, name: str) -> None:
    res = pd.DataFrame(search.cv_results_)
    for i, row in res.iterrows():
        with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
            mlflow.set_tag("candidate_rank", int(row["rank_test_roc_auc"]))
            mlflow.log_params(_clean_params(row["params"]))
            mlflow.log_metrics({f"cv_{m}": float(row[f"mean_test_{m}"]) for m in evaluate.SCORING})


def train_family(name, estimator, grid, X_tr, y_tr, X_te, y_te, cv, out_dir: Path) -> dict:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "model_selection"})
        t0 = time.time()
        search = GridSearchCV(build_pipeline(estimator), grid, cv=cv, scoring=evaluate.SCORING,
                              refit="roc_auc", n_jobs=-1, return_train_score=False)
        search.fit(X_tr, y_tr)
        _log_candidates(search, name)

        best = search.best_estimator_
        i = search.best_index_
        cv_metrics = {f"cv_{m}": float(search.cv_results_[f"mean_test_{m}"][i])
                      for m in evaluate.SCORING}
        cv_std = {f"cv_{m}_std": float(search.cv_results_[f"std_test_{m}"][i])
                  for m in evaluate.SCORING}
        proba = best.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_metrics = {f"test_{k}": float(v)
                        for k, v in evaluate.compute_metrics(y_te, pred, proba).items()}

        mlflow.log_params(_clean_params(search.best_params_))
        mlflow.log_params({"cv_folds": cv.get_n_splits(), "n_train": len(X_tr),
                           "n_test": len(X_te), "grid_size": len(search.cv_results_["params"])})
        mlflow.log_metrics({**cv_metrics, **cv_std, **test_metrics,
                            "train_seconds": time.time() - t0})

        fam_dir = out_dir / name
        fam_dir.mkdir(parents=True, exist_ok=True)
        evaluate.plot_confusion(y_te, pred, f"{name} - test confusion matrix",
                                fam_dir / "confusion_matrix.png")
        evaluate.plot_roc(y_te, proba, f"{name} - test ROC", fam_dir / "roc_curve.png")
        fi = feature_importance(best)
        if fi is not None:
            evaluate.plot_feature_importance(*fi, f"{name} - feature importance",
                                             fam_dir / "feature_importance.png")
        pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
            fam_dir / "cv_results.csv", index=False)
        mlflow.log_artifacts(str(fam_dir), artifact_path="evaluation")

        signature = infer_signature(X_tr, best.predict_proba(X_tr)[:, 1])
        mlflow.sklearn.log_model(best, name="model", signature=signature,
                                 input_example=X_tr.head(3),
                                 serialization_format="cloudpickle")
        log.info("%-20s cv_roc_auc=%.4f test_roc_auc=%.4f test_acc=%.4f", name,
                 cv_metrics["cv_roc_auc"], test_metrics["test_roc_auc"],
                 test_metrics["test_accuracy"])
        return {"name": name, "run_id": run.info.run_id, "pipeline": best,
                "best_params": search.best_params_, **cv_metrics, **cv_std, **test_metrics}


# Lower = simpler / more interpretable. Used by the one-standard-error rule.
COMPLEXITY = {"logistic_regression": 0, "knn": 1, "svm_rbf": 2, "random_forest": 3,
              "xgboost": 4}


def select_champion(results: list[dict]) -> dict:
    """One-standard-error rule.

    Take the family with the highest mean CV ROC-AUC, then pick the *simplest*
    family whose CV ROC-AUC is within one standard deviation of it. With ~240
    training rows, differences smaller than the fold-to-fold noise are not
    meaningful, so the simpler, more interpretable model wins ties.
    """
    top = max(results, key=lambda r: r["cv_roc_auc"])
    cutoff = top["cv_roc_auc"] - top["cv_roc_auc_std"]
    eligible = [r for r in results if r["cv_roc_auc"] >= cutoff]
    champ = min(eligible, key=lambda r: (COMPLEXITY.get(r["name"], 99), -r["cv_roc_auc"]))
    champ["selection_rule"] = (f"1-SE rule: best={top['name']} {top['cv_roc_auc']:.4f}, "
                               f"cutoff={cutoff:.4f}, eligible={[r['name'] for r in eligible]}")
    return champ


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny grids (CI smoke test)")
    parser.add_argument("--data", default=str(CLEAN_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    df = load_clean(Path(args.data))
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (est, grid) in model_space(args.quick).items():
            results.append(train_family(name, est, grid, X_tr, y_tr, X_te, y_te, cv, Path(tmp)))
            # keep a copy of plots for the report
            for f in (Path(tmp) / name).glob("*.png"):
                dest = FIG_DIR / "models" / name
                dest.mkdir(parents=True, exist_ok=True)
                (dest / f.name).write_bytes(f.read_bytes())

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pipeline", "selection_rule")}
                            for r in results])
    summary = summary.sort_values("cv_roc_auc", ascending=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIG_DIR.parent / "model_comparison.csv", index=False)
    best = select_champion(results)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)
    meta = {
        "model_name": best["name"], "mlflow_run_id": best["run_id"],
        "best_params": _clean_params(best["best_params"]),
        "features": FEATURES, "threshold": 0.5,
        "selection_rule": best["selection_rule"],
        "cv_roc_auc": best["cv_roc_auc"], "test_roc_auc": best["test_roc_auc"],
        "test_accuracy": best["test_accuracy"], "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_name="champion-selection"):
        mlflow.set_tags({"stage": "champion", "selection_rule": best["selection_rule"]})
        mlflow.log_params({"champion": best["name"], "champion_run_id": best["run_id"]})
        mlflow.log_metrics({"champion_cv_roc_auc": best["cv_roc_auc"],
                            "champion_test_roc_auc": best["test_roc_auc"]})
        mlflow.log_artifact(str(FIG_DIR.parent / "model_comparison.csv"))
        mlflow.log_artifact(str(MODEL_META_PATH))
        try:
            mlflow.register_model(f"runs:/{best['run_id']}/model", "heart-disease-classifier")
        except Exception as exc:  # registry unavailable on some backends
            log.warning("Model registry skipped: %s", exc)

    log.info("Champion: %s (cv_roc_auc=%.4f) -> %s", best["name"], best["cv_roc_auc"], MODEL_PATH)
    print(summary[["name", "cv_roc_auc", "cv_accuracy", "test_roc_auc", "test_accuracy",
                   "test_precision", "test_recall"]].round(4).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
