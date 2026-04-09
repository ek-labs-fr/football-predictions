"""SHAP explainability for tree-based models (Step 3.8)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

ARTEFACTS_DIR = Path("artefacts")
OUTPUTS_DIR = Path("outputs")


def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    model_name: str = "home",
) -> shap.Explanation:
    """Compute SHAP values for a tree-based model.

    Parameters
    ----------
    model:
        A fitted tree model (XGBoost, LightGBM).
    X:
        Feature matrix to explain.
    model_name:
        Label for logging.

    Returns
    -------
    shap.Explanation
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("Computed SHAP values for %s model (%d samples)", model_name, len(X))
    return shap_values


def save_shap_artefacts(
    explainer_home: shap.TreeExplainer,
    shap_values_home: shap.Explanation,
    feature_names: list[str],
    artefacts_dir: Path = ARTEFACTS_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Save SHAP explainer and feature importance to disk."""
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(explainer_home, artefacts_dir / "shap_explainer.pkl")

    # Feature importance (mean absolute SHAP value)
    mean_abs = np.abs(shap_values_home.values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    importance_df.to_csv(artefacts_dir / "shap_feature_importance.csv", index=False)

    logger.info("Saved SHAP artefacts to %s", artefacts_dir)


def generate_shap_plots(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Generate and save SHAP plots.

    Saves summary, bar importance, and top-5 dependence plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Summary dot plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(outputs_dir / "shap_summary_home_win.png", dpi=150)
    plt.close()

    # Bar importance
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(outputs_dir / "shap_bar_importance.png", dpi=150)
    plt.close()

    # Dependence plots for top 5 features
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[-5:][::-1]
    for idx in top5_idx:
        feat = X.columns[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values.values, X, show=False)
        plt.tight_layout()
        plt.savefig(outputs_dir / f"shap_dependence_{feat}.png", dpi=150)
        plt.close()

    logger.info("Saved SHAP plots to %s", outputs_dir)
