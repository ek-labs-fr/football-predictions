"""Rolling xG features for the club pipeline.

Mirrors the structure of ``src.features.rolling`` but consumes per-fixture xG
(from the match-statistics CSV) instead of goals. Produces, per
``(fixture_id, team_id)``:

    xg_for_avg_l10        mean of xG-for over the team's prior 10 matches
    xg_against_avg_l10    mean of xG-against over the team's prior 10 matches

Pre-2023 fixtures have no xG in API-Football's coverage of PL / La Liga /
Ligue 1, so windows that contain too few populated priors return NaN — the
training-table layer fills those with the column median.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features import io

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

_WINDOW_LONG = 10
_MIN_PRIORS = 3


def _team_xg_history(fixtures: pd.DataFrame, match_stats: pd.DataFrame) -> pd.DataFrame:
    """Reshape per-fixture xG into one row per (team, match)."""
    fx = fixtures[["fixture_id", "date", "home_team_id", "away_team_id"]].copy()
    fx["date"] = pd.to_datetime(fx["date"], utc=True, errors="coerce")
    keep = ["fixture_id", "home_xg", "away_xg"]
    stats = match_stats[[c for c in keep if c in match_stats.columns]].copy()
    fx = fx.merge(stats, on="fixture_id", how="left")

    home = pd.DataFrame(
        {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"],
            "team_id": fx["home_team_id"],
            "xg_for": fx.get("home_xg"),
            "xg_against": fx.get("away_xg"),
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"],
            "team_id": fx["away_team_id"],
            "xg_for": fx.get("away_xg"),
            "xg_against": fx.get("home_xg"),
        }
    )
    history = pd.concat([home, away], ignore_index=True)
    return history.sort_values(["team_id", "date"]).reset_index(drop=True)


def _rolling_prior(s: pd.Series) -> pd.Series:
    """Mean of the prior _WINDOW_LONG values, requiring >= _MIN_PRIORS populated."""
    return s.shift(1).rolling(_WINDOW_LONG, min_periods=_MIN_PRIORS).mean()


def compute_xg_rolling_features(
    fixtures_path: str | Path = PROCESSED_DIR / "all_fixtures_club.csv",
    match_stats_path: str | Path = PROCESSED_DIR / "match_statistics_club.csv",
    output_path: str | Path = PROCESSED_DIR / "features_xg_rolling_club.csv",
) -> pd.DataFrame:
    """Compute xg_for_avg_l10 / xg_against_avg_l10 per (fixture_id, team_id).

    Uses prior matches only (``shift(1)``) to prevent leakage. Returns the
    written DataFrame.
    """
    if not io.exists(match_stats_path):
        logger.warning(
            "Match-stats CSV not found at %s — writing empty xG-rolling output",
            match_stats_path,
        )
        empty = pd.DataFrame(
            columns=[
                "fixture_id",
                "team_id",
                "xg_for_avg_l10",
                "xg_against_avg_l10",
            ]
        )
        io.write_csv(output_path, empty)
        return empty

    fixtures = io.read_csv(fixtures_path)
    match_stats = io.read_csv(match_stats_path)

    history = _team_xg_history(fixtures, match_stats)
    history["xg_for_avg_l10"] = history.groupby("team_id")["xg_for"].transform(_rolling_prior)
    history["xg_against_avg_l10"] = history.groupby("team_id")["xg_against"].transform(
        _rolling_prior,
    )

    out = history[
        [
            "fixture_id",
            "team_id",
            "xg_for_avg_l10",
            "xg_against_avg_l10",
        ]
    ].copy()

    coverage = out["xg_for_avg_l10"].notna().mean() if len(out) else 0.0
    io.write_csv(output_path, out)
    logger.info(
        "xG rolling features: %d (fixture, team) rows -> %s (coverage %.1f%%)",
        len(out),
        output_path,
        coverage * 100,
    )
    return out
