"""Batch fixture prediction core: upcoming + holdout, per competition.

For each configured competition (WC 2026, Premier League, La Liga, Ligue 1):
    * upcoming  — predictions on the inference table for league_id
    * past      — predictions on the training-table holdout (with actuals merged)

Outputs go to S3 (when DATA_BUCKET is set) under:
    web/data/competitions.json
    web/data/upcoming_<competition>.json
    web/data/past_<competition>.json

The legacy combined CSV/Parquet files (predictions_{national_wc2026,club})
are still emitted for debugging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from src.features import io
from src.models.calibrate import _bivariate_poisson_matrix
from src.models.train import _make_holdout_masks, get_feature_columns

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Mode + competition registry
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ModeConfig:
    training_table: str
    inference_table: str
    artefacts_prefix: str
    legacy_csv: str
    legacy_parquet: str


MODES: dict[str, ModeConfig] = {
    "national": ModeConfig(
        training_table="data/processed/training_table.csv",
        inference_table="data/processed/inference_table.parquet",
        artefacts_prefix="artefacts",
        legacy_csv="outputs/predictions_national_wc2026.csv",
        legacy_parquet="outputs/predictions_national_wc2026.parquet",
    ),
    "club": ModeConfig(
        training_table="data/processed/training_table_club.csv",
        inference_table="data/processed/inference_table_club.parquet",
        artefacts_prefix="artefacts/club",
        legacy_csv="outputs/predictions_club.csv",
        legacy_parquet="outputs/predictions_club.parquet",
    ),
}


@dataclass(frozen=True)
class Competition:
    id: str
    name: str
    mode: str
    league_id: int
    past_label: str


COMPETITIONS: list[Competition] = [
    Competition("wc-2026", "FIFA World Cup 2026", "national", 1, "Holdout: World Cup 2022"),
    Competition("premier-league", "Premier League", "club", 39, "Holdout: 2024–25 season"),
    Competition("la-liga", "La Liga", "club", 140, "Holdout: 2024–25 season"),
    Competition("ligue-1", "Ligue 1", "club", 61, "Holdout: 2024–25 season"),
]


# ------------------------------------------------------------------
# Artefact loading
# ------------------------------------------------------------------


def _load_pickle(key: str) -> object:
    return joblib.load(BytesIO(io.read_bytes(key)))


def _load_artefacts(prefix: str) -> tuple[object, object, object | None, float]:
    model_home = _load_pickle(f"{prefix}/model_final_home.pkl")
    model_away = _load_pickle(f"{prefix}/model_final_away.pkl")
    scaler_key = f"{prefix}/model_final_scaler.pkl"
    scaler = _load_pickle(scaler_key) if io.exists(scaler_key) else None
    rho = float(io.read_json(f"{prefix}/rho.json")["rho"])
    return model_home, model_away, scaler, rho


# ------------------------------------------------------------------
# Prediction primitive
# ------------------------------------------------------------------


_OUTCOMES = ("home_win", "draw", "away_win")


def _predict_rows(
    rows: pd.DataFrame,
    feature_cols: list[str],
    medians: pd.Series,
    model_home: object,
    model_away: object,
    scaler: object | None,
    rho: float,
) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in rows.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing[:5]}...")

    X = rows[feature_cols].fillna(medians)
    X_input = scaler.transform(X) if scaler is not None else X.values

    lh = np.clip(model_home.predict(X_input), 0.01, 10.0)
    la = np.clip(model_away.predict(X_input), 0.01, 10.0)

    scores: list[str] = []
    p_h: list[float] = []
    p_d: list[float] = []
    p_a: list[float] = []
    for h, a in zip(lh, la, strict=True):
        mat = _bivariate_poisson_matrix(h, a, rho)
        idx = np.unravel_index(mat.argmax(), mat.shape)
        scores.append(f"{int(idx[0])}-{int(idx[1])}")
        p_h.append(float(np.tril(mat, -1).sum()))
        p_d.append(float(np.trace(mat)))
        p_a.append(float(np.triu(mat, 1).sum()))

    out = rows.copy()
    out["lambda_home"] = np.round(lh, 3)
    out["lambda_away"] = np.round(la, 3)
    out["predicted_score"] = scores
    out["p_home_win"] = np.round(p_h, 4)
    out["p_draw"] = np.round(p_d, 4)
    out["p_away_win"] = np.round(p_a, 4)
    probs = np.column_stack([p_h, p_d, p_a])
    out["predicted_outcome"] = [_OUTCOMES[i] for i in probs.argmax(axis=1)]
    return out


# ------------------------------------------------------------------
# Mode-level: upcoming + holdout DataFrames
# ------------------------------------------------------------------


def _ensure_train_df_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def predict_upcoming(mode: str) -> pd.DataFrame:
    cfg = MODES[mode]
    train_df = io.read_csv(cfg.training_table)
    feature_cols = get_feature_columns(train_df, mode=mode)
    medians = train_df[feature_cols].median()

    model_home, model_away, scaler, rho = _load_artefacts(cfg.artefacts_prefix)
    inf = io.read_parquet(cfg.inference_table)

    out = _predict_rows(inf, feature_cols, medians, model_home, model_away, scaler, rho)
    out = out.sort_values("date").reset_index(drop=True)

    legacy = out[
        [c for c in [
            "fixture_id", "date", "league_id", "round",
            "home_team_name", "away_team_name",
            "lambda_home", "lambda_away", "predicted_score",
            "p_home_win", "p_draw", "p_away_win",
        ] if c in out.columns]
    ].rename(columns={"predicted_score": "most_likely_score"})
    io.write_csv(cfg.legacy_csv, legacy)
    io.write_parquet(cfg.legacy_parquet, legacy)

    logger.info("[%s] upcoming: %d rows (rho=%.4f)", mode, len(out), rho)
    return out


def predict_holdout(mode: str) -> pd.DataFrame:
    cfg = MODES[mode]
    train_df = _ensure_train_df_dates(io.read_csv(cfg.training_table))
    feature_cols = get_feature_columns(train_df, mode=mode)
    medians = train_df[feature_cols].median()

    _train_mask, test_mask = _make_holdout_masks(train_df, mode)
    holdout = train_df[test_mask].copy()

    model_home, model_away, scaler, rho = _load_artefacts(cfg.artefacts_prefix)
    out = _predict_rows(holdout, feature_cols, medians, model_home, model_away, scaler, rho)

    out["actual_home_goals"] = out["home_goals"].astype(int)
    out["actual_away_goals"] = out["away_goals"].astype(int)
    out["actual_score"] = (
        out["actual_home_goals"].astype(str) + "-" + out["actual_away_goals"].astype(str)
    )
    out["actual_outcome"] = np.where(
        out["actual_home_goals"] > out["actual_away_goals"], "home_win",
        np.where(
            out["actual_home_goals"] < out["actual_away_goals"], "away_win", "draw",
        ),
    )
    out["correct_outcome"] = out["predicted_outcome"] == out["actual_outcome"]
    out["correct_score"] = out["predicted_score"] == out["actual_score"]
    out = out.sort_values("date").reset_index(drop=True)

    logger.info("[%s] holdout: %d rows", mode, len(out))
    return out


# ------------------------------------------------------------------
# Per-competition splitting + JSON shape
# ------------------------------------------------------------------


_UPCOMING_COLS = [
    "fixture_id", "date", "round",
    "home_team_name", "away_team_name",
    "predicted_score",
    "lambda_home", "lambda_away",
    "p_home_win", "p_draw", "p_away_win",
    "predicted_outcome",
]

_PAST_EXTRA_COLS = [
    "actual_home_goals", "actual_away_goals", "actual_score",
    "actual_outcome", "correct_outcome", "correct_score",
]


def _to_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    cols = [c for c in columns if c in df.columns]
    sub = df[cols].copy()
    if "date" in sub.columns:
        sub["date"] = pd.to_datetime(sub["date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    for c in sub.select_dtypes(include="bool").columns:
        sub[c] = sub[c].astype(bool)
    return sub.to_dict(orient="records")


def _performance(past: pd.DataFrame) -> dict[str, float | int]:
    if len(past) == 0:
        return {
            "total_matches": 0,
            "correct_outcomes": 0,
            "correct_scores": 0,
            "outcome_accuracy": 0.0,
            "score_accuracy": 0.0,
            "mae_avg": 0.0,
        }
    mae_home = float((past["actual_home_goals"] - past["lambda_home"]).abs().mean())
    mae_away = float((past["actual_away_goals"] - past["lambda_away"]).abs().mean())
    return {
        "total_matches": int(len(past)),
        "correct_outcomes": int(past["correct_outcome"].sum()),
        "correct_scores": int(past["correct_score"].sum()),
        "outcome_accuracy": round(float(past["correct_outcome"].mean()), 4),
        "score_accuracy": round(float(past["correct_score"].mean()), 4),
        "mae_avg": round((mae_home + mae_away) / 2, 4),
    }


def _filter_competition(df: pd.DataFrame, league_id: int) -> pd.DataFrame:
    if "league_id" not in df.columns:
        return df.iloc[0:0]
    return df[df["league_id"] == league_id].copy()


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


def publish_dashboard_json() -> dict[str, object]:
    """Run upcoming + holdout for both modes, then write per-competition JSON."""
    upcoming_by_mode = {m: predict_upcoming(m) for m in MODES}
    past_by_mode = {m: predict_holdout(m) for m in MODES}

    summary: dict[str, object] = {"competitions": []}

    manifest = []
    for comp in COMPETITIONS:
        upcoming_df = _filter_competition(upcoming_by_mode[comp.mode], comp.league_id)
        past_df = _filter_competition(past_by_mode[comp.mode], comp.league_id)

        upcoming_payload = {
            "competition_id": comp.id,
            "competition_name": comp.name,
            "matches": _to_records(upcoming_df, _UPCOMING_COLS),
        }
        past_payload = {
            "competition_id": comp.id,
            "competition_name": comp.name,
            "label": comp.past_label,
            "matches": _to_records(past_df, _UPCOMING_COLS + _PAST_EXTRA_COLS),
            "performance": _performance(past_df),
        }

        io.write_json(f"web/data/upcoming_{comp.id}.json", upcoming_payload)
        io.write_json(f"web/data/past_{comp.id}.json", past_payload)

        manifest.append({
            "id": comp.id,
            "name": comp.name,
            "mode": comp.mode,
            "past_label": comp.past_label,
            "upcoming_count": len(upcoming_payload["matches"]),
            "past_count": len(past_payload["matches"]),
        })
        summary["competitions"].append({
            "id": comp.id,
            "upcoming": len(upcoming_payload["matches"]),
            "past": len(past_payload["matches"]),
            "outcome_accuracy": past_payload["performance"]["outcome_accuracy"],
        })

    io.write_json("web/data/competitions.json", manifest)
    logger.info("Published dashboard JSON for %d competitions", len(manifest))
    return summary


# Backward-compat shim for the existing CLI script (predict_inference.py).
def predict_mode(mode: str) -> dict[str, object]:
    df = predict_upcoming(mode)
    return {"mode": mode, "rows": len(df)}


CONFIGS = MODES  # legacy alias
