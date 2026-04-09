"""Feature selection pipeline (Step 3.5).

Sequential stages: variance threshold → correlation filter → RFECV → permutation importance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd  # noqa: TCH002
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ARTEFACTS_DIR = Path("artefacts")
OUTPUTS_DIR = Path("outputs")


def _variance_threshold(X: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Remove features with variance below threshold."""
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    removed = [c for c, keep in zip(X.columns, sel.get_support(), strict=True) if not keep]
    logger.info("Variance threshold: removed %d features: %s", len(removed), removed)
    return [c for c in X.columns if c not in removed]


def _correlation_filter(X: pd.DataFrame, threshold: float = 0.90) -> list[str]:
    """Remove one from each pair with |correlation| > threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        if high:
            to_drop.add(col)
    logger.info("Correlation filter: removed %d features: %s", len(to_drop), sorted(to_drop))
    return [c for c in X.columns if c not in to_drop]


def _rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    n_splits: int = 5,
    min_features: int = 10,
) -> list[str]:
    """Recursive feature elimination with cross-validation."""
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
    cv = TimeSeriesSplit(n_splits=n_splits)
    rfecv = RFECV(
        estimator=lr,
        step=1,
        cv=cv,
        scoring="neg_log_loss",
        min_features_to_select=min_features,
        n_jobs=-1,
    )
    rfecv.fit(X, y, sample_weight=sample_weight)  # type: ignore[call-arg]
    selected = X.columns[rfecv.support_].tolist()
    logger.info("RFECV: selected %d/%d features", len(selected), len(X.columns))
    return selected


def _permutation_filter(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 30,
) -> list[str]:
    """Drop features with negative permutation importance."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=42)
    importances = result.importances_mean
    keep = [c for c, imp in zip(X.columns, importances, strict=True) if imp >= 0]
    removed = [c for c, imp in zip(X.columns, importances, strict=True) if imp < 0]
    logger.info("Permutation filter: removed %d features: %s", len(removed), removed)
    return keep


def run_feature_selection(
    X_train: pd.DataFrame,
    y_outcome: pd.Series,
    sample_weight: pd.Series,
    xgb_model: object | None = None,
    artefacts_dir: Path = ARTEFACTS_DIR,
) -> list[str]:
    """Run the full feature selection pipeline.

    Parameters
    ----------
    X_train:
        Training features.
    y_outcome:
        Encoded outcome labels.
    sample_weight:
        Sample weights.
    xgb_model:
        Optional trained XGBoost model for permutation importance.
    artefacts_dir:
        Where to save selected feature list.

    Returns
    -------
    list[str]
        Selected feature names.
    """
    logger.info("Starting feature selection: %d features", len(X_train.columns))

    # Stage 1: Variance threshold
    cols = _variance_threshold(X_train)
    X = X_train[cols]

    # Stage 2: Correlation filter
    cols = _correlation_filter(X)
    X = X_train[cols]

    # Stage 3: RFECV
    cols = _rfecv(X, y_outcome, sample_weight)
    X = X_train[cols]

    # Stage 4: Permutation importance (if model provided)
    if xgb_model is not None:
        cols = _permutation_filter(xgb_model, X, y_outcome)

    logger.info("Final selected features (%d): %s", len(cols), cols)

    # Save
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cols, artefacts_dir / "selected_features.pkl")

    return cols
