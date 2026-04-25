"""Batch fixture prediction core, used by both the CLI and the Lambda handler.

Loads model_final + rho.json from the artefacts prefix, the training table
(for feature columns + medians), and the inference table — then writes a
predictions CSV/Parquet under the predictions prefix.

I/O always goes through src.features.io, so the same code runs locally
(DATA_BUCKET unset) and in Lambda (DATA_BUCKET set, S3 backend).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.features import io
from src.models.calibrate import _bivariate_poisson_matrix
from src.models.train import get_feature_columns

logger = logging.getLogger(__name__)


def _wc_group_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["league_id"] == 1].copy()


@dataclass(frozen=True)
class ModeConfig:
    training_table: str
    inference_table: str
    artefacts_prefix: str
    output_csv: str
    output_parquet: str
    row_filter: Callable[[pd.DataFrame], pd.DataFrame]


CONFIGS: dict[str, ModeConfig] = {
    "national": ModeConfig(
        training_table="data/processed/training_table.csv",
        inference_table="data/processed/inference_table.parquet",
        artefacts_prefix="artefacts",
        output_csv="outputs/predictions_national_wc2026.csv",
        output_parquet="outputs/predictions_national_wc2026.parquet",
        row_filter=_wc_group_only,
    ),
    "club": ModeConfig(
        training_table="data/processed/training_table_club.csv",
        inference_table="data/processed/inference_table_club.parquet",
        artefacts_prefix="artefacts/club",
        output_csv="outputs/predictions_club.csv",
        output_parquet="outputs/predictions_club.parquet",
        row_filter=lambda df: df.copy(),
    ),
}

OUTPUT_BASE_COLS = [
    "fixture_id",
    "date",
    "league_id",
    "round",
    "home_team_name",
    "away_team_name",
]


def _load_pickle(key: str) -> object:
    return joblib.load(BytesIO(io.read_bytes(key)))


def _load_artefacts(prefix: str) -> tuple[object, object, object | None, float]:
    model_home = _load_pickle(f"{prefix}/model_final_home.pkl")
    model_away = _load_pickle(f"{prefix}/model_final_away.pkl")
    scaler_key = f"{prefix}/model_final_scaler.pkl"
    scaler = _load_pickle(scaler_key) if io.exists(scaler_key) else None
    rho = float(io.read_json(f"{prefix}/rho.json")["rho"])
    return model_home, model_away, scaler, rho


def predict_mode(mode: str) -> dict[str, object]:
    """Run prediction for one mode. Returns a summary dict (counts + paths)."""
    cfg = CONFIGS[mode]

    train_df = io.read_csv(cfg.training_table)
    feature_cols = get_feature_columns(train_df, mode=mode)
    medians = train_df[feature_cols].median()

    model_home, model_away, scaler, rho = _load_artefacts(cfg.artefacts_prefix)

    inf = cfg.row_filter(io.read_parquet(cfg.inference_table))

    missing = [c for c in feature_cols if c not in inf.columns]
    if missing:
        raise KeyError(f"[{mode}] inference table missing feature columns: {missing[:5]}...")

    X = inf[feature_cols].fillna(medians)
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

    cols = [c for c in OUTPUT_BASE_COLS if c in inf.columns]
    out = inf[cols].copy()
    out["lambda_home"] = np.round(lh, 3)
    out["lambda_away"] = np.round(la, 3)
    out["most_likely_score"] = scores
    out["p_home_win"] = np.round(p_h, 4)
    out["p_draw"] = np.round(p_d, 4)
    out["p_away_win"] = np.round(p_a, 4)
    out = out.sort_values("date").reset_index(drop=True)

    io.write_csv(cfg.output_csv, out)
    io.write_parquet(cfg.output_parquet, out)

    logger.info(
        "[%s] %d fixtures predicted (rho=%.4f) -> %s, %s",
        mode, len(out), rho, cfg.output_csv, cfg.output_parquet,
    )
    return {
        "mode": mode,
        "rows": len(out),
        "rho": rho,
        "csv": cfg.output_csv,
        "parquet": cfg.output_parquet,
    }
