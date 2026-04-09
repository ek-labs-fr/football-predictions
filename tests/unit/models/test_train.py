"""Tests for model training utilities."""

from __future__ import annotations

from src.models.train import (
    most_likely_score,
    outcome_probs_from_lambdas,
    predict_match,
    scoreline_matrix,
)


class TestScorelineMatrix:
    def test_shape(self) -> None:
        mat = scoreline_matrix(1.5, 1.2, max_goals=5)
        assert mat.shape == (6, 6)

    def test_sums_to_one(self) -> None:
        mat = scoreline_matrix(1.5, 1.2)
        assert abs(mat.sum() - 1.0) < 0.01

    def test_probabilities_non_negative(self) -> None:
        mat = scoreline_matrix(1.5, 1.2)
        assert (mat >= 0).all()


class TestOutcomeProbsFromLambdas:
    def test_sums_to_one(self) -> None:
        probs = outcome_probs_from_lambdas(1.5, 1.2)
        assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-6

    def test_strong_home_favours_home(self) -> None:
        probs = outcome_probs_from_lambdas(3.0, 0.5)
        assert probs["home_win"] > probs["draw"]
        assert probs["home_win"] > probs["away_win"]

    def test_equal_lambdas(self) -> None:
        probs = outcome_probs_from_lambdas(1.5, 1.5)
        assert abs(probs["home_win"] - probs["away_win"]) < 0.01


class TestMostLikelyScore:
    def test_low_scoring(self) -> None:
        score = most_likely_score(0.8, 0.6)
        # With low lambdas, 0-0 or 1-0 is most likely
        parts = score.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_high_scoring(self) -> None:
        score = most_likely_score(3.0, 2.5)
        parts = score.split("-")
        h, a = int(parts[0]), int(parts[1])
        # Should predict something reasonable
        assert h >= 1
        assert a >= 1


class TestPredictMatch:
    def test_output_format(self) -> None:
        result = predict_match(1.5, 1.2)
        assert "lambda_home" in result
        assert "lambda_away" in result
        assert "most_likely_score" in result
        assert "home_win" in result
        assert "draw" in result
        assert "away_win" in result

    def test_probs_sum_to_one(self) -> None:
        result = predict_match(1.5, 1.2)
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_lambdas_preserved(self) -> None:
        result = predict_match(2.0, 0.8)
        assert result["lambda_home"] == 2.0
        assert result["lambda_away"] == 0.8
