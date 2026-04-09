"""Tests for Monte Carlo tournament simulation."""

from __future__ import annotations

import numpy as np

from src.models.simulate import (
    simulate_group_stage,
    simulate_knockout_match,
    simulate_match,
)


class TestSimulateMatch:
    def test_returns_ints(self) -> None:
        rng = np.random.default_rng(42)
        hg, ag = simulate_match(1.5, 1.2, rng=rng)
        assert isinstance(hg, int)
        assert isinstance(ag, int)
        assert hg >= 0
        assert ag >= 0

    def test_deterministic_with_seed(self) -> None:
        r1 = simulate_match(1.5, 1.2, rng=np.random.default_rng(99))
        r2 = simulate_match(1.5, 1.2, rng=np.random.default_rng(99))
        assert r1 == r2


class TestSimulateKnockoutMatch:
    def test_always_has_winner(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            hg, ag, decided = simulate_knockout_match(1.0, 1.0, rng=rng)
            assert hg != ag  # there must be a winner
            assert decided in ("FT", "AET", "PEN")

    def test_decided_in_values(self) -> None:
        rng = np.random.default_rng(42)
        decisions = set()
        for _ in range(200):
            _, _, decided = simulate_knockout_match(1.0, 1.0, rng=rng)
            decisions.add(decided)
        # With 200 attempts at equal lambdas, we should see all outcomes
        assert "FT" in decisions
        # AET or PEN should also appear
        assert len(decisions) >= 2


class TestSimulateGroupStage:
    def test_probabilities_sum(self) -> None:
        teams = [1, 2, 3, 4]

        def get_lambdas(home: int, away: int) -> tuple[float, float]:
            return 1.2, 1.0

        result = simulate_group_stage(teams, get_lambdas, n_sims=500, rng=np.random.default_rng(42))

        assert len(result) == 4
        assert "advance_prob" in result.columns
        # All advance probs should be between 0 and 1
        assert (result["advance_prob"] >= 0).all()
        assert (result["advance_prob"] <= 1).all()
        # Total advance prob should be ~2 (top 2 advance out of 4)
        assert abs(result["advance_prob"].sum() - 2.0) < 0.1
