"""Probability calibration with bivariate Poisson correlation (Step 3.7).

Fits a correlation parameter ρ to correct draw probabilities by adjusting
the independent Poisson assumption.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from src.models.train import SplitData, TrainedModel, predict_lambdas

logger = logging.getLogger(__name__)

ARTEFACTS_DIR = Path("artefacts")
_MAX_GOALS = 10


def _bivariate_poisson_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = _MAX_GOALS,
) -> np.ndarray:
    """Compute bivariate Poisson scoreline matrix with correlation ρ.

    Uses the diagonal-inflation method: the independent Poisson matrix is
    adjusted by shifting probability mass to/from the diagonal (draws).
    """
    # Independent Poisson
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    mat = np.outer(home_probs, away_probs)

    # Apply correlation: inflate diagonal by ρ factor
    if rho != 0:
        for i in range(max_goals + 1):
            mat[i, i] *= 1 + rho

        # Renormalize
        mat = mat / mat.sum()

    return mat


def outcome_probs_bivariate(lambda_home: float, lambda_away: float, rho: float) -> dict[str, float]:
    """Derive outcome probabilities from bivariate Poisson."""
    mat = _bivariate_poisson_matrix(lambda_home, lambda_away, rho)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    total = p_home + p_draw + p_away
    return {"home_win": p_home / total, "draw": p_draw / total, "away_win": p_away / total}


def fit_rho(
    model: TrainedModel,
    split: SplitData,
    calibration_frac: float = 0.15,
) -> float:
    """Fit the bivariate Poisson correlation parameter ρ on a calibration set.

    Uses the last calibration_frac of the training data as calibration set.
    Minimizes the Brier score of draw predictions.
    """
    n = len(split.X_train)
    cal_start = int(n * (1 - calibration_frac))
    X_cal = split.X_train.iloc[cal_start:]
    y_home_cal = split.y_home_train.iloc[cal_start:].values
    y_away_cal = split.y_away_train.iloc[cal_start:].values

    lh, la = predict_lambdas(model, X_cal, split)

    # Actual draw indicator
    is_draw = (y_home_cal == y_away_cal).astype(float)

    def loss(rho: float) -> float:
        brier_sum = 0.0
        for i in range(len(X_cal)):
            probs = outcome_probs_bivariate(lh[i], la[i], rho)
            brier_sum += (probs["draw"] - is_draw[i]) ** 2
        return brier_sum / len(X_cal)

    result = minimize_scalar(loss, bounds=(-0.5, 0.5), method="bounded")
    rho = float(result.x)
    logger.info("Fitted ρ = %.4f (Brier loss = %.4f)", rho, result.fun)
    return rho


def save_calibration(
    model: TrainedModel,
    rho: float,
    artefacts_dir: Path = ARTEFACTS_DIR,
) -> None:
    """Save calibrated model artefacts."""
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    if model.model_home is not None:
        joblib.dump(model.model_home, artefacts_dir / "model_home_calibrated.pkl")
    if model.model_away is not None:
        joblib.dump(model.model_away, artefacts_dir / "model_away_calibrated.pkl")
    if model.scaler is not None:
        joblib.dump(model.scaler, artefacts_dir / "scaler.pkl")

    rho_path = artefacts_dir / "rho.json"
    rho_path.write_text(json.dumps({"rho": rho}, indent=2), encoding="utf-8")
    logger.info("Saved calibration artefacts to %s (ρ=%.4f)", artefacts_dir, rho)
