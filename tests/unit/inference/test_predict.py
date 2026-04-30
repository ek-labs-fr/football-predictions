"""Tests for the inference scoreline decision rule."""

from __future__ import annotations

from src.inference.predict import _expected_scoreline


class TestExpectedScoreline:
    def test_clear_home_win_lifts_off_one_zero(self) -> None:
        assert _expected_scoreline(1.55, 0.77, "home_win") == (2, 1)

    def test_strong_home_favorite(self) -> None:
        assert _expected_scoreline(2.24, 0.56, "home_win") == (2, 1)

    def test_close_match_predicted_home_bumps_to_two_one(self) -> None:
        # round(1.21)=1, round(1.05)=1; bumped because predicted home_win
        assert _expected_scoreline(1.21, 1.05, "home_win") == (2, 1)

    def test_clear_away_win(self) -> None:
        assert _expected_scoreline(0.77, 1.55, "away_win") == (1, 2)

    def test_predicted_draw_with_unequal_lambdas_takes_average(self) -> None:
        # round(1.21)=1, round(1.05)=1 — same after rounding, stays 1-1
        assert _expected_scoreline(1.21, 1.05, "draw") == (1, 1)

    def test_predicted_draw_when_rounded_disagree_uses_mean(self) -> None:
        # round(1.6)=2, round(1.0)=1; predicted draw → average rounds to 1
        assert _expected_scoreline(1.6, 1.0, "draw") == (1, 1)

    def test_low_scoring_draw(self) -> None:
        assert _expected_scoreline(0.4, 0.3, "draw") == (0, 0)

    def test_outcome_consistency_when_round_equal_but_home_win(self) -> None:
        # both round to 1, predicted home win → bump home to 2
        assert _expected_scoreline(0.9, 1.1, "home_win") == (2, 1)

    def test_outcome_consistency_when_round_equal_but_away_win(self) -> None:
        assert _expected_scoreline(1.1, 0.9, "away_win") == (1, 2)

    def test_caps_at_max_predicted_goals(self) -> None:
        h, a = _expected_scoreline(15.0, 0.5, "home_win")
        assert h == 10
        assert a <= 10
