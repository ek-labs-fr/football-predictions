"""Report prediction accuracy bucketed by model + decision-rule lineage.

Reads frozen predictions from ``predictions/<fid>.json``, joins with actuals
from the training tables, and prints a table grouped by
``decision_rule_version`` and ``model_trained_at``. Useful for answering
questions like "did rounded_expected_v1 outperform argmax_v0?" or
"is the new retrain producing better predictions?"

Predictions written before lineage stamping (pre-PR #10) won't have those
fields and are bucketed as ``unstamped`` / ``unknown``.

Usage:
    uv run python scripts/prediction_lineage_report.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_FINISHED = {"FT", "AET", "PEN"}
_TRAINING_TABLES = (
    "data/processed/training_table.csv",
    "data/processed/training_table_club.csv",
)


def _load_predictions() -> pd.DataFrame:
    keys = [k for k in io.list_keys("predictions/") if k.endswith(".json")]
    rows = []
    for k in keys:
        try:
            payload = io.read_json(k)
        except (OSError, ValueError):
            continue
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "decision_rule_version" not in df.columns:
        df["decision_rule_version"] = "unstamped"
    else:
        df["decision_rule_version"] = df["decision_rule_version"].fillna("unstamped")
    if "model_trained_at" not in df.columns:
        df["model_trained_at"] = "unknown"
    else:
        df["model_trained_at"] = df["model_trained_at"].fillna("unknown")
    return df


def _load_actuals() -> pd.DataFrame:
    frames = []
    for tbl in _TRAINING_TABLES:
        if not io.exists(tbl):
            continue
        df = io.read_csv(tbl)
        df = df[df["status"].isin(_FINISHED)][
            ["fixture_id", "home_goals", "away_goals"]
        ].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["fixture_id", "home_goals", "away_goals"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("fixture_id")
    out["fixture_id"] = out["fixture_id"].astype(int)
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    return out


def _score_outcomes(joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.copy()
    joined["actual_score"] = (
        joined["home_goals"].astype(str) + "-" + joined["away_goals"].astype(str)
    )
    joined["actual_outcome"] = pd.Series(
        ["home_win"] * len(joined), index=joined.index,
    )
    joined.loc[joined["home_goals"] < joined["away_goals"], "actual_outcome"] = "away_win"
    joined.loc[joined["home_goals"] == joined["away_goals"], "actual_outcome"] = "draw"
    joined["correct_outcome"] = joined["predicted_outcome"] == joined["actual_outcome"]
    joined["correct_score"] = joined["predicted_score"] == joined["actual_score"]
    joined["abs_h_err"] = (joined["lambda_home"] - joined["home_goals"]).abs()
    joined["abs_a_err"] = (joined["lambda_away"] - joined["away_goals"]).abs()
    return joined


def report() -> pd.DataFrame:
    preds = _load_predictions()
    if preds.empty:
        logger.warning("No frozen predictions found under predictions/")
        return pd.DataFrame()

    actuals = _load_actuals()
    if actuals.empty:
        logger.warning("No actuals found in training tables")
        return pd.DataFrame()

    preds["fixture_id"] = preds["fixture_id"].astype(int)
    joined = preds.merge(actuals, on="fixture_id", how="inner")
    if joined.empty:
        logger.warning("No frozen predictions matched any FT fixture")
        return pd.DataFrame()

    scored = _score_outcomes(joined)
    grouped = (
        scored.groupby(["decision_rule_version", "model_trained_at"], dropna=False)
        .agg(
            n=("fixture_id", "count"),
            outcome_acc=("correct_outcome", "mean"),
            score_acc=("correct_score", "mean"),
            mae_home=("abs_h_err", "mean"),
            mae_away=("abs_a_err", "mean"),
        )
        .reset_index()
        .sort_values(["decision_rule_version", "model_trained_at"])
    )
    grouped["outcome_acc"] = grouped["outcome_acc"].round(4)
    grouped["score_acc"] = grouped["score_acc"].round(4)
    grouped["mae_home"] = grouped["mae_home"].round(3)
    grouped["mae_away"] = grouped["mae_away"].round(3)
    return grouped


def main() -> None:
    df = report()
    if df.empty:
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    out_path = Path("outputs/prediction_lineage_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
