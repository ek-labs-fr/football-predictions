"""Verify the xG-retrained club model on the same holdout the experiment used.

Compares the new production artefacts against the pre-xG snapshot in
``artefacts/club_v0_pre_xg/`` so we can confirm dispersion + accuracy moved
the right way before the change is committed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.calibrate import _bivariate_poisson_matrix, fit_rho
from src.models.train import (
    _make_holdout_masks,
    create_split,
    get_feature_columns,
    predict_lambdas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEW_TABLE = "data/processed/training_table_club.csv"
OLD_TABLE = "data/processed/training_table_club_pre_xg.csv"
NEW_DIR = Path("artefacts/club")
OLD_DIR = Path("artefacts/club_v0_pre_xg")


def _load_artefact(d: Path):
    home = joblib.load(d / "model_final_home.pkl")
    away = joblib.load(d / "model_final_away.pkl")
    scaler_path = d / "model_final_scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    import json
    rho = json.loads((d / "rho.json").read_text())["rho"]
    return home, away, scaler, float(rho)


def evaluate(label: str, table_path: str, artefacts: Path) -> dict:
    df = pd.read_csv(table_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    train_mask, test_mask = _make_holdout_masks(df, "club")
    train_df = df[train_mask]
    test_df = df[test_mask]
    feature_cols = get_feature_columns(df, mode="club")
    medians = train_df[feature_cols].median()
    X_test = test_df[feature_cols].fillna(medians)

    home, away, scaler, rho = _load_artefact(artefacts)
    X_input = scaler.transform(X_test) if scaler is not None else X_test.values
    lh = np.clip(home.predict(X_input), 0.01, 10.0)
    la = np.clip(away.predict(X_input), 0.01, 10.0)

    actual_h = test_df["home_goals"].astype(int).to_numpy()
    actual_a = test_df["away_goals"].astype(int).to_numpy()

    correct_score = []
    correct_outcome = []
    for i, (h, a) in enumerate(zip(lh, la, strict=True)):
        mat = _bivariate_poisson_matrix(h, a, rho)
        si, sj = np.unravel_index(mat.argmax(), mat.shape)
        correct_score.append(int(si == actual_h[i] and sj == actual_a[i]))
        ph = float(np.tril(mat, -1).sum())
        pd_ = float(np.trace(mat))
        pa = float(np.triu(mat, 1).sum())
        marg = int(np.argmax([ph, pd_, pa]))
        actual_o = (
            0 if actual_h[i] > actual_a[i]
            else 1 if actual_h[i] == actual_a[i]
            else 2
        )
        correct_outcome.append(int(marg == actual_o))

    return {
        "label": label,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_cols),
        "rho": round(rho, 4),
        "lh_std": round(float(lh.std()), 3),
        "la_std": round(float(la.std()), 3),
        "lh_p90": round(float(np.quantile(lh, 0.9)), 2),
        "la_p90": round(float(np.quantile(la, 0.9)), 2),
        "high_total_share": round(float(((lh + la) >= 3.5).mean()), 3),
        "exact_score_acc": round(float(np.mean(correct_score)), 3),
        "outcome_acc": round(float(np.mean(correct_outcome)), 3),
        "mae_h": round(float(np.abs(actual_h - lh).mean()), 3),
        "mae_a": round(float(np.abs(actual_a - la).mean()), 3),
    }


def main() -> None:
    rows = []
    if Path(OLD_TABLE).exists() and OLD_DIR.exists():
        rows.append(evaluate("pre_xg (snapshot)", OLD_TABLE, OLD_DIR))
    rows.append(evaluate("with_xg (production)", NEW_TABLE, NEW_DIR))
    df = pd.DataFrame(rows)
    print("\n=== Production retrain verification (clubs holdout) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
