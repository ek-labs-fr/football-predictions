"""Production readiness checklist (Step 3.11).

Usage:
    uv run python scripts/validate_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARTEFACTS = Path("artefacts")


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def validate() -> bool:
    all_ok = True

    print("\n=== Production Readiness Checklist ===\n")

    # Model artefacts exist
    all_ok &= _check(
        "Calibrated home model exists",
        (ARTEFACTS / "model_home_calibrated.pkl").exists(),
    )
    all_ok &= _check(
        "Calibrated away model exists",
        (ARTEFACTS / "model_away_calibrated.pkl").exists(),
    )
    all_ok &= _check("Scaler exists", (ARTEFACTS / "scaler.pkl").exists())

    # Rho
    rho_path = ARTEFACTS / "rho.json"
    if rho_path.exists():
        rho_data = json.loads(rho_path.read_text(encoding="utf-8"))
        rho = rho_data.get("rho", None)
        all_ok &= _check("ρ fitted", rho is not None, f"ρ = {rho}")
    else:
        _check("rho.json exists", False)
        all_ok = False

    # Selected features
    all_ok &= _check(
        "Selected features saved",
        (ARTEFACTS / "selected_features.pkl").exists(),
    )

    # SHAP
    all_ok &= _check(
        "SHAP explainer saved",
        (ARTEFACTS / "shap_explainer.pkl").exists(),
    )
    all_ok &= _check(
        "SHAP feature importance saved",
        (ARTEFACTS / "shap_feature_importance.csv").exists(),
    )

    # Best params
    all_ok &= _check(
        "Best hyperparameters saved",
        (ARTEFACTS / "best_params.json").exists(),
    )

    # Model comparison
    all_ok &= _check(
        "Model comparison table saved",
        (Path("outputs") / "model_comparison.csv").exists(),
    )

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
