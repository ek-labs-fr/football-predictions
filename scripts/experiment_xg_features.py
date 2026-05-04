"""Test whether adding xG rolling features widens the predicted goal range.

Builds xg_for_avg_l10 / xg_against_avg_l10 from the cached /fixtures/statistics
responses and compares Poisson Linear and LightGBM Poisson trained with vs
without xG, on the same clubs holdout.

Coverage: pre-2023 has no xG, 2023+ has full xG. The model uses median
imputation for older training rows; xG signal is learned from 2023 forward.

Usage:
    uv run python scripts/experiment_xg_features.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.calibrate import _bivariate_poisson_matrix, fit_rho
from src.models.train import (
    create_split,
    predict_lambdas,
    train_baselines,
    train_candidates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw/club/fixtures_statistics")
TRAINING_BASELINE = "data/processed/training_table_club.csv"
TRAINING_AUGMENTED = "data/processed/training_table_club_xg.csv"
FIXTURES_CLUB = "data/processed/all_fixtures_club.csv"

_WINDOW = 10
_MIN_PRIORS = 3
_TARGETS = ("poisson_linear", "lightgbm_poisson")


def parse_xg_from_cache() -> pd.DataFrame:
    """Walk cached /fixtures/statistics JSONs, extract (fixture_id, home_xg, away_xg)."""
    files = list(CACHE_DIR.glob("*.json"))
    logger.info("Parsing %d cached statistics files", len(files))
    rows: list[dict] = []
    for i, f in enumerate(files):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        params = data.get("parameters", {}) or {}
        fid_raw = params.get("fixture")
        if fid_raw is None:
            continue
        try:
            fid = int(fid_raw)
        except (TypeError, ValueError):
            continue
        response = data.get("response", []) or []
        if len(response) < 2:
            continue
        xgs: list[float | None] = []
        for team_block in response[:2]:
            stats = team_block.get("statistics", []) or []
            xg: float | None = None
            for s in stats:
                if s.get("type") == "expected_goals":
                    v = s.get("value")
                    if v is None:
                        break
                    try:
                        xg = float(v)
                    except (TypeError, ValueError):
                        xg = None
                    break
            xgs.append(xg)
        if xgs[0] is None and xgs[1] is None:
            continue
        rows.append({"fixture_id": fid, "home_xg": xgs[0], "away_xg": xgs[1]})
        if (i + 1) % 2000 == 0:
            logger.info("  ...%d/%d files (%d rows so far)", i + 1, len(files), len(rows))
    df = pd.DataFrame(rows).drop_duplicates(subset=["fixture_id"])
    logger.info("Extracted xG for %d fixtures", len(df))
    return df


def compute_xg_rolling(fixtures: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Per (fixture_id, team_id), compute xg_for_avg_l10 and xg_against_avg_l10.

    Uses prior matches only (shift(1) before rolling) to prevent leakage.
    Requires >= _MIN_PRIORS xG-populated priors in the window or returns NaN.
    """
    fx = fixtures[["fixture_id", "date", "home_team_id", "away_team_id"]].copy()
    fx = fx.merge(xg_df, on="fixture_id", how="left")
    fx["date"] = pd.to_datetime(fx["date"], utc=True, errors="coerce")

    # Reshape: one row per (team, match)
    home = pd.DataFrame(
        {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"],
            "team_id": fx["home_team_id"],
            "xg_for": fx["home_xg"],
            "xg_against": fx["away_xg"],
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": fx["fixture_id"],
            "date": fx["date"],
            "team_id": fx["away_team_id"],
            "xg_for": fx["away_xg"],
            "xg_against": fx["home_xg"],
        }
    )
    history = pd.concat([home, away], ignore_index=True)
    history = history.sort_values(["team_id", "date"]).reset_index(drop=True)

    def rolling_prior(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(_WINDOW, min_periods=_MIN_PRIORS).mean()

    grouped = history.groupby("team_id")
    history["xg_for_avg_l10"] = grouped["xg_for"].transform(rolling_prior)
    history["xg_against_avg_l10"] = grouped["xg_against"].transform(rolling_prior)

    coverage = history["xg_for_avg_l10"].notna().mean()
    logger.info("Rolling xG coverage: %.1f%% of (team, match) rows", coverage * 100)

    return history[["fixture_id", "team_id", "xg_for_avg_l10", "xg_against_avg_l10"]]


def augment_training_table(rolling_xg: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_BASELINE)

    home_xg = rolling_xg.rename(
        columns={
            "team_id": "home_team_id",
            "xg_for_avg_l10": "home_xg_for_avg_l10",
            "xg_against_avg_l10": "home_xg_against_avg_l10",
        }
    )
    away_xg = rolling_xg.rename(
        columns={
            "team_id": "away_team_id",
            "xg_for_avg_l10": "away_xg_for_avg_l10",
            "xg_against_avg_l10": "away_xg_against_avg_l10",
        }
    )
    df = df.merge(home_xg, on=["fixture_id", "home_team_id"], how="left")
    df = df.merge(away_xg, on=["fixture_id", "away_team_id"], how="left")

    new_cols = [
        "home_xg_for_avg_l10",
        "home_xg_against_avg_l10",
        "away_xg_for_avg_l10",
        "away_xg_against_avg_l10",
    ]
    pop = df[new_cols].notna().mean()
    logger.info("Training-table xG feature populated rate:\n%s", pop.to_string())

    return df


def evaluate(label: str, model_name: str, table_path: str) -> dict:
    """Train candidates on table_path, return holdout metrics for model_name."""
    split = create_split(training_table_path=table_path, mode="club")
    baselines = train_baselines(split)
    candidates = train_candidates(split)
    by_name = {m.name: m for m in baselines + candidates}
    if model_name not in by_name:
        raise RuntimeError(f"{model_name!r} not in trained models")

    model = by_name[model_name]
    rho = fit_rho(model, split)
    lh, la = predict_lambdas(model, split.X_test, split)

    actual_h = split.y_home_test.to_numpy()
    actual_a = split.y_away_test.to_numpy()
    correct_score: list[int] = []
    correct_outcome: list[int] = []
    for i, (h, a) in enumerate(zip(lh, la, strict=True)):
        mat = _bivariate_poisson_matrix(h, a, rho)
        si, sj = np.unravel_index(mat.argmax(), mat.shape)
        correct_score.append(int(si == actual_h[i] and sj == actual_a[i]))
        ph = float(np.tril(mat, -1).sum())
        pd_ = float(np.trace(mat))
        pa = float(np.triu(mat, 1).sum())
        marg = int(np.argmax([ph, pd_, pa]))
        actual_o = 0 if actual_h[i] > actual_a[i] else 1 if actual_h[i] == actual_a[i] else 2
        correct_outcome.append(int(marg == actual_o))

    return {
        "label": label,
        "model": model_name,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "n_features": len(split.feature_cols),
        "rho": round(rho, 4),
        "lh_std": round(float(lh.std()), 3),
        "la_std": round(float(la.std()), 3),
        "high_total_share": round(float(((lh + la) >= 3.5).mean()), 3),
        "lh_p90": round(float(np.quantile(lh, 0.9)), 2),
        "la_p90": round(float(np.quantile(la, 0.9)), 2),
        "exact_score_acc": round(float(np.mean(correct_score)), 3),
        "outcome_acc": round(float(np.mean(correct_outcome)), 3),
        "mae_h": round(float(np.abs(actual_h - lh).mean()), 3),
        "mae_a": round(float(np.abs(actual_a - la).mean()), 3),
    }


def main() -> None:
    logger.info("Step 1: parse xG from cache")
    xg_df = parse_xg_from_cache()

    logger.info("Step 2: compute rolling xG features")
    fixtures = pd.read_csv(FIXTURES_CLUB)
    rolling_xg = compute_xg_rolling(fixtures, xg_df)

    logger.info("Step 3: augment training table")
    augmented = augment_training_table(rolling_xg)
    Path(TRAINING_AUGMENTED).parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(TRAINING_AUGMENTED, index=False)
    logger.info("Wrote augmented table -> %s (%d rows)", TRAINING_AUGMENTED, len(augmented))

    logger.info("Step 4: train + evaluate")
    rows: list[dict] = []
    for model_name in _TARGETS:
        for label, path in (("baseline", TRAINING_BASELINE), ("with_xg", TRAINING_AUGMENTED)):
            logger.info("=> %s @ %s", model_name, label)
            rows.append(evaluate(label, model_name, path))

    print("\n=== Results (clubs holdout 2024-25) ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n=== Side-by-side per model ===")
    for model_name in _TARGETS:
        a = next(r for r in rows if r["model"] == model_name and r["label"] == "baseline")
        b = next(r for r in rows if r["model"] == model_name and r["label"] == "with_xg")
        print(f"\n{model_name}:")
        for key in (
            "n_features",
            "lh_std",
            "la_std",
            "lh_p90",
            "la_p90",
            "high_total_share",
            "exact_score_acc",
            "outcome_acc",
            "mae_h",
            "mae_a",
        ):
            delta = b[key] - a[key]
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<20} baseline={a[key]:>8}   with_xg={b[key]:>8}   delta={sign}{delta}")


if __name__ == "__main__":
    main()
