"""Tests for the inference version-stamping behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import io as feature_io
from src.inference.predict import _DECISION_RULE_VERSION, _store_prediction


class TestVersionStamping:
    def _row(self) -> pd.Series:
        return pd.Series({
            "fixture_id": 12345,
            "lambda_home": 1.55,
            "lambda_away": 0.77,
            "predicted_score": "1-0",
            "p_home_win": 0.55,
            "p_draw": 0.27,
            "p_away_win": 0.18,
            "predicted_outcome": "home_win",
        })

    def test_decision_rule_version_constant(self) -> None:
        assert _DECISION_RULE_VERSION == "argmax_v0"

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

        assert payload["decision_rule_version"] == "argmax_v0"
        assert payload["model_trained_at"] == "2026-04-26T20:41:12+00:00"
        assert payload["fixture_id"] == 12345

        on_disk = feature_io.read_json("predictions/12345.json")
        assert on_disk["decision_rule_version"] == "argmax_v0"
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
        assert payload["decision_rule_version"] == "argmax_v0"


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
        assert "T" in ts
        assert len(ts) == 25

    def test_returns_none_for_missing_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        monkeypatch.delenv("DATA_BUCKET", raising=False)
        feature_io._client.cache_clear()

        assert feature_io.last_modified(tmp_path / "missing.pkl") is None
