"""Diagnose whether the clubs model's goal range is too narrow.

Two possible failure modes:
  A) lambda values are too compressed around the league mean (model can't tell
     a Man City vs Sheffield Utd mismatch from a midtable match).
  B) lambda values are well-spread but the Poisson conditional distribution
     itself is too narrow (no over-dispersion).

This script compares actual goal distribution vs predicted lambda distribution
to attribute the gap.

Usage:
    uv run python scripts/probe_goal_dispersion.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from src.inference.predict import predict_holdout


def main() -> None:
    df = predict_holdout("club", decision_rule="argmax_v0")
    n = len(df)

    actual_h = df["actual_home_goals"].to_numpy()
    actual_a = df["actual_away_goals"].to_numpy()
    lam_h = df["lambda_home"].to_numpy()
    lam_a = df["lambda_away"].to_numpy()

    # Per-side: actual vs lambda
    print("=== Per-side dispersion check ===")
    header = (
        f"{'side':<8} {'mean':>7} {'std':>7} {'min':>6} {'q10':>6} {'q50':>6} {'q90':>6} {'max':>6}"
    )
    print(header)
    for label, vals in [
        ("home act", actual_h),
        ("home lam", lam_h),
        ("away act", actual_a),
        ("away lam", lam_a),
    ]:
        q = np.quantile(vals, [0.10, 0.50, 0.90])
        print(
            f"{label:<8} {vals.mean():>7.2f} {vals.std():>7.2f} "
            f"{vals.min():>6.2f} {q[0]:>6.2f} {q[1]:>6.2f} {q[2]:>6.2f} {vals.max():>6.2f}",
        )

    # Goal-count histograms (per side, 0..6+)
    print("\n=== Per-side goal distribution: actual vs Poisson(lambda) implied ===")
    print(f"{'goals':>5} {'home_act':>9} {'home_pred':>10} {'away_act':>9} {'away_pred':>10}")
    for k in range(7):
        # Actual share scoring exactly k
        act_h_share = float((actual_h == k).mean()) if k < 6 else float((actual_h >= k).mean())
        act_a_share = float((actual_a == k).mean()) if k < 6 else float((actual_a >= k).mean())
        # Predicted = mean over fixtures of Poisson PMF at k under each lambda
        if k < 6:
            pred_h = float(poisson.pmf(k, lam_h).mean())
            pred_a = float(poisson.pmf(k, lam_a).mean())
        else:
            pred_h = float((1 - poisson.cdf(k - 1, lam_h)).mean())
            pred_a = float((1 - poisson.cdf(k - 1, lam_a)).mean())
        label = f"{k}+" if k == 6 else str(k)
        print(
            f"{label:>5} {act_h_share:>9.1%} {pred_h:>10.1%} {act_a_share:>9.1%} {pred_a:>10.1%}",
        )

    # How often is lambda >= 2 (a fixture where the model "expects" 2+ goals)?
    print("\n=== How often does the model predict a high-scoring side? ===")
    for thresh in [1.5, 2.0, 2.5, 3.0]:
        share_h_lam = float((lam_h >= thresh).mean())
        share_h_act = float((actual_h >= thresh).mean())
        share_a_lam = float((lam_a >= thresh).mean())
        share_a_act = float((actual_a >= thresh).mean())
        print(
            f"  >= {thresh}:  home  pred {share_h_lam:>5.1%}  vs actual {share_h_act:>5.1%}  "
            f"|  away  pred {share_a_lam:>5.1%}  vs actual {share_a_act:>5.1%}",
        )

    # How often does the joint Poisson actually concentrate above 1-goal cells?
    n_hi_lam = int(((lam_h + lam_a) >= 3.5).sum())
    pct_hi_lam = n_hi_lam / n
    print(f"\nFixtures with lambda_h + lambda_a >= 3.5:  {n_hi_lam} / {n} ({pct_hi_lam:.1%})")
    n_high_actual = int(((actual_h + actual_a) >= 4).sum())
    pct_high_act = n_high_actual / n
    print(f"Fixtures with actual goals  >= 4:           {n_high_actual} / {n} ({pct_high_act:.1%})")


if __name__ == "__main__":
    main()
