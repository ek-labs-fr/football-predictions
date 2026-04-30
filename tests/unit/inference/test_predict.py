"""Tests for the inference scoreline decision rule and version stamping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.predict import (
    _DECISION_RULE_VERSION,
    _expected_scoreline,
    _store_prediction,
)
from src.features import io as feature_io


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


class TestVersionStamping:
    def _row(self) -> pd.Series:
        return pd.Series({
            "fixture_id": 12345,
            "lambda_home": 1.55,
            "lambda_away": 0.77,
            "predicted_score": "2-1",
            "p_home_win": 0.55,
            "p_draw": 0.27,
            "p_away_win": 0.18,
            "predicted_outcome": "home_win",
        })

    def test_decision_rule_version_constant(self) -> None:
        assert _DECISION_RULE_VERSION == "rounded_expected_v1"

    def test_store_writes_decision_rule_and_trained_at(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DATA_BUCKET", raising=False)
        feature_io._client.cache_clear()

        payload = _store_prediction(
            fid=12345,
            prediction_row=self._row(),
            backfill=False,
            model_trained_at="2026-04-26T20:41:12+00:00",
        )

        assert payload["decision_rule_version"] == "rounded_expected_v1"
        assert payload["model_trained_at"] == "2026-04-26T20:41:12+00:00"
        assert payload["fixture_id"] == 12345

        on_disk = feature_io.read_json("predictions/12345.json")
        assert on_disk["decision_rule_version"] == "rounded_expected_v1"
        assert on_disk["model_trained_at"] == "2026-04-26T20:41:12+00:00"

    def test_store_handles_missing_trained_at(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DATA_BUCKET", raising=False)
        feature_io._client.cache_clear()

        payload = _store_prediction(
            fid=99999,
            prediction_row=self._row(),
            backfill=False,
            model_trained_at=None,
        )

        assert payload["model_trained_at"] is None
        assert payload["decision_rule_version"] == "rounded_expected_v1"


class TestLastModified:
    def test_returns_iso_timestamp_for_existing_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.delenv("DATA_BUCKET", raising=False)
        feature_io._client.cache_clear()

        f = tmp_path / "model.pkl"
        f.write_bytes(b"x")
        ts = feature_io.last_modified(f)

        assert ts is not None
        assert ts.endswith("+00:00")
        # ISO format: YYYY-MM-DDTHH:MM:SS+00:00
        assert "T" in ts
        assert len(ts) == 25

    def test_returns_none_for_missing_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.delenv("DATA_BUCKET", raising=False)
        feature_io._client.cache_clear()

        assert feature_io.last_modified(tmp_path / "missing.pkl") is None
