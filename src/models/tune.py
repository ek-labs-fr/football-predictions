"""Hyperparameter tuning with Optuna (Step 3.6)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

ARTEFACTS_DIR = Path("artefacts")


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    sample_weight: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Optuna objective: mean MAE across CV folds for home+away Poisson models."""
    params = {
        "objective": "count:poisson",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbosity": 0,
    }

    cv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []

    for train_idx, val_idx in cv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        yh_tr, yh_val = y_home[train_idx], y_home[val_idx]
        ya_tr, ya_val = y_away[train_idx], y_away[val_idx]
        w_tr = sample_weight[train_idx]

        model_h = XGBRegressor(**params)
        model_a = XGBRegressor(**params)
        model_h.fit(X_tr, yh_tr, sample_weight=w_tr)
        model_a.fit(X_tr, ya_tr, sample_weight=w_tr)

        pred_h = model_h.predict(X_val)
        pred_a = model_a.predict(X_val)

        mae_h = mean_absolute_error(yh_val, np.round(pred_h))
        mae_a = mean_absolute_error(ya_val, np.round(pred_a))
        mae_scores.append((mae_h + mae_a) / 2)

    return float(np.mean(mae_scores))


def tune_xgboost(
    X: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    sample_weight: np.ndarray,
    n_trials: int = 100,
    timeout: int = 3600,
    artefacts_dir: Path = ARTEFACTS_DIR,
) -> dict:  # type: ignore[type-arg]
    """Run Optuna hyperparameter search for XGBoost Poisson.

    Parameters
    ----------
    X, y_home, y_away, sample_weight:
        Training data arrays.
    n_trials:
        Number of Optuna trials.
    timeout:
        Maximum time in seconds.
    artefacts_dir:
        Where to save best_params.json.

    Returns
    -------
    dict
        Best hyperparameters.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", study_name="xgboost_poisson_tune")

    study.optimize(
        lambda trial: _objective(trial, X, y_home, y_away, sample_weight),
        n_trials=n_trials,
        timeout=timeout,
    )

    best = study.best_params
    logger.info("Best trial: MAE=%.4f, params=%s", study.best_value, best)

    artefacts_dir.mkdir(parents=True, exist_ok=True)
    path = artefacts_dir / "best_params.json"
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    logger.info("Saved best params to %s", path)

    return best
