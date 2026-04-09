"""Model evaluation: metrics, comparison tables, and diagnostic plots.

Step 3.4 — Rigorous comparison of all models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from src.models.train import (
    SplitData,
    TrainedModel,
    predict_lambdas,
    predict_outcome_probs,
)

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------


def exact_scoreline_accuracy(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> float:
    """Fraction of matches where the most likely predicted scoreline matches actual."""
    correct = 0
    for yh, ya, lh, la in zip(y_home, y_away, lambda_home, lambda_away, strict=True):
        from src.models.train import scoreline_matrix

        mat = scoreline_matrix(lh, la)
        pred_h, pred_a = np.unravel_index(mat.argmax(), mat.shape)
        if pred_h == yh and pred_a == ya:
            correct += 1
    return correct / len(y_home) if len(y_home) > 0 else 0.0


def ranked_probability_score(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    """Compute mean Ranked Probability Score for 3-class outcome.

    Lower is better. Measures calibration across ordered categories.
    Classes are assumed ordered: away_win(0), draw(1), home_win(2).
    """
    n = len(y_true_idx)
    if n == 0:
        return 0.0
    rps_sum = 0.0
    for i in range(n):
        true_one_hot = np.zeros(3)
        true_one_hot[int(y_true_idx[i])] = 1.0
        cum_pred = np.cumsum(probs[i])
        cum_true = np.cumsum(true_one_hot)
        rps_sum += np.sum((cum_pred - cum_true) ** 2) / 2.0
    return rps_sum / n


def compute_goal_metrics(
    y_home: np.ndarray,
    y_away: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    y_outcome_idx: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Compute all goal-level and derived outcome metrics."""
    mae_home = mean_absolute_error(y_home, np.round(lambda_home))
    mae_away = mean_absolute_error(y_away, np.round(lambda_away))
    mae_avg = (mae_home + mae_away) / 2

    esa = exact_scoreline_accuracy(y_home, y_away, lambda_home, lambda_away)
    rps = ranked_probability_score(y_outcome_idx, probs)

    pred_outcome = probs.argmax(axis=1)
    acc = accuracy_score(y_outcome_idx, pred_outcome)

    # Log loss — clip probabilities for numerical stability
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    probs_normed = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    ll = log_loss(y_outcome_idx, probs_normed, labels=[0, 1, 2])

    # Brier score (multi-class: mean of per-class Brier scores)
    brier = 0.0
    for cls in range(3):
        y_bin = (y_outcome_idx == cls).astype(float)
        brier += brier_score_loss(y_bin, probs_normed[:, cls])
    brier /= 3

    return {
        "mae_home": round(mae_home, 4),
        "mae_away": round(mae_away, 4),
        "mae_avg": round(mae_avg, 4),
        "exact_scoreline_acc": round(esa, 4),
        "rps": round(rps, 4),
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
    }


def compute_classifier_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Compute outcome-only metrics for classifier models."""
    pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, pred)

    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    probs_normed = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    ll = log_loss(y_true, probs_normed, labels=[0, 1, 2])

    brier = 0.0
    for cls in range(3):
        y_bin = (y_true == cls).astype(float)
        brier += brier_score_loss(y_bin, probs_normed[:, cls])
    brier /= 3

    rps = ranked_probability_score(y_true, probs_normed)

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "rps": round(rps, 4),
    }


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------


def evaluate_model(
    model: TrainedModel,
    split: SplitData,
) -> dict[str, Any]:
    """Evaluate a single model on the holdout test set."""
    if model.is_poisson:
        lh, la = predict_lambdas(model, split.X_test, split)
        probs = predict_outcome_probs(model, split.X_test, split)
        metrics = compute_goal_metrics(
            split.y_home_test.values,
            split.y_away_test.values,
            lh,
            la,
            split.y_outcome_test.values,
            probs,
        )
    else:
        probs = predict_outcome_probs(model, split.X_test, split)
        metrics = compute_classifier_metrics(split.y_outcome_test.values, probs)

    metrics["model"] = model.name
    metrics["type"] = "poisson" if model.is_poisson else "classifier"
    metrics["train_time_s"] = round(model.train_time, 2)

    return metrics


def evaluate_all(
    models: list[TrainedModel],
    split: SplitData,
    output_path: Path = OUTPUTS_DIR / "model_comparison.csv",
) -> pd.DataFrame:
    """Evaluate all models and produce a comparison table."""
    results = [evaluate_model(m, split) for m in models]
    df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Model comparison saved to %s", output_path)
    logger.info("\n%s", df.to_string(index=False))

    return df


def get_confusion_matrix(
    model: TrainedModel,
    split: SplitData,
) -> np.ndarray:
    """Compute confusion matrix for a model on the test set."""
    probs = predict_outcome_probs(model, split.X_test, split)
    pred = probs.argmax(axis=1)
    return confusion_matrix(split.y_outcome_test.values, pred, labels=[0, 1, 2])


def get_classification_report(
    model: TrainedModel,
    split: SplitData,
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1."""
    probs = predict_outcome_probs(model, split.X_test, split)
    pred = probs.argmax(axis=1)
    p, r, f1, sup = precision_recall_fscore_support(
        split.y_outcome_test.values, pred, labels=[0, 1, 2], zero_division=0
    )
    classes = split.label_encoder.classes_
    return {
        classes[i]: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f1[i], 4)}
        for i in range(len(classes))
    }
