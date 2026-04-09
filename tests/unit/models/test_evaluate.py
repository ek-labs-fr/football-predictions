"""Tests for model evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.models.evaluate import (
    compute_classifier_metrics,
    compute_goal_metrics,
    exact_scoreline_accuracy,
    ranked_probability_score,
)


class TestExactScorelineAccuracy:
    def test_perfect_prediction(self) -> None:
        # Lambda that makes 1-0 most likely
        acc = exact_scoreline_accuracy(
            np.array([1]), np.array([0]), np.array([1.2]), np.array([0.4])
        )
        assert acc == 1.0

    def test_wrong_prediction(self) -> None:
        # Predict low-scoring but actual is 5-4
        acc = exact_scoreline_accuracy(
            np.array([5]), np.array([4]), np.array([0.5]), np.array([0.3])
        )
        assert acc == 0.0

    def test_empty(self) -> None:
        acc = exact_scoreline_accuracy(np.array([]), np.array([]), np.array([]), np.array([]))
        assert acc == 0.0


class TestRankedProbabilityScore:
    def test_perfect_prediction(self) -> None:
        # Predict home_win (class 2) with probability 1.0, actual is home_win
        rps = ranked_probability_score(np.array([2]), np.array([[0.0, 0.0, 1.0]]))
        assert rps == 0.0

    def test_worst_prediction(self) -> None:
        # Predict away_win with 100%, actual is home_win
        rps = ranked_probability_score(np.array([2]), np.array([[1.0, 0.0, 0.0]]))
        assert rps > 0

    def test_empty(self) -> None:
        rps = ranked_probability_score(np.array([]), np.array([]).reshape(0, 3))
        assert rps == 0.0


class TestComputeGoalMetrics:
    def test_returns_all_keys(self) -> None:
        metrics = compute_goal_metrics(
            y_home=np.array([1, 2, 0]),
            y_away=np.array([0, 1, 1]),
            lambda_home=np.array([1.2, 1.8, 0.5]),
            lambda_away=np.array([0.5, 1.0, 1.2]),
            y_outcome_idx=np.array([2, 2, 0]),
            probs=np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]),
        )
        assert "mae_avg" in metrics
        assert "exact_scoreline_acc" in metrics
        assert "rps" in metrics
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "brier_score" in metrics


class TestComputeClassifierMetrics:
    def test_returns_all_keys(self) -> None:
        metrics = compute_classifier_metrics(
            y_true=np.array([0, 1, 2]),
            probs=np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]),
        )
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "brier_score" in metrics
        assert "rps" in metrics
        assert metrics["accuracy"] == 1.0
