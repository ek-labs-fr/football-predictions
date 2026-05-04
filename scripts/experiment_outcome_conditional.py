"""One-off experiment: compare argmax_v0 vs outcome_conditional_v0 decision rules.

Runs the production club model (Poisson Linear) on the 2024-25 holdout under
both decision rules and reports scoreline diversity + accuracy. λ values and
outcome marginals are identical between rules — only the served scoreline
changes.

Does NOT touch production artefacts. Reads the existing club training table
and the production club artefacts.

Usage:
    uv run python scripts/experiment_outcome_conditional.py
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.inference.predict import COMPETITIONS, _load_artefacts, predict_holdout
from src.models.calibrate import _bivariate_poisson_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_RULES = ("argmax_v0", "outcome_conditional_v0")
_LOW_SCORE_CELLS = {"0-0", "1-0", "0-1", "1-1"}
_LEAGUES = {c.league_id: c.name for c in COMPETITIONS if c.mode == "club"}


def _scoreline_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    counts = Counter(df["predicted_score"])
    return {
        "n": n,
        "unique_scorelines": len(counts),
        "share_1_1": counts.get("1-1", 0) / n,
        "share_low": sum(c for s, c in counts.items() if s in _LOW_SCORE_CELLS) / n,
        "exact_score_acc": float(df["correct_score"].mean()),
        "outcome_acc": float(df["correct_outcome"].mean()),
        "mae_home": float((df["actual_home_goals"] - df["lambda_home"]).abs().mean()),
        "mae_away": float((df["actual_away_goals"] - df["lambda_away"]).abs().mean()),
        "top10": counts.most_common(10),
    }


def _print_block(label: str, summaries: dict[str, dict]) -> None:
    s_a = summaries["argmax_v0"]
    s_b = summaries["outcome_conditional_v0"]
    print(f"\n=== {label}  (n={s_a['n']}) ===")
    print(f"{'metric':<22} {'argmax_v0':>14} {'outcome_cond_v0':>18} {'delta':>10}")

    def row(metric: str, a: float, b: float, fmt: str = "{:.3f}") -> None:
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(
            f"{metric:<22} {fmt.format(a):>14} {fmt.format(b):>18} "
            f"{sign}{fmt.format(delta):>9}"
        )

    row("share 1-1", s_a["share_1_1"], s_b["share_1_1"], "{:.1%}")
    row("share low (4 cells)", s_a["share_low"], s_b["share_low"], "{:.1%}")
    row("unique scorelines", s_a["unique_scorelines"], s_b["unique_scorelines"], "{:d}")
    row("exact score accuracy", s_a["exact_score_acc"], s_b["exact_score_acc"], "{:.1%}")
    row("outcome accuracy", s_a["outcome_acc"], s_b["outcome_acc"], "{:.1%}")
    row("MAE home", s_a["mae_home"], s_b["mae_home"], "{:.3f}")
    row("MAE away", s_a["mae_away"], s_b["mae_away"], "{:.3f}")

    print("\nTop 10 predicted scorelines:")
    print(f"  {'argmax_v0':<28}  {'outcome_conditional_v0':<28}")
    for i in range(10):
        a = s_a["top10"][i] if i < len(s_a["top10"]) else (("", 0))
        b = s_b["top10"][i] if i < len(s_b["top10"]) else (("", 0))
        a_str = f"{a[0]}: {a[1]} ({a[1] / s_a['n']:.1%})" if a[0] else ""
        b_str = f"{b[0]}: {b[1]} ({b[1] / s_b['n']:.1%})" if b[0] else ""
        print(f"  {a_str:<28}  {b_str:<28}")


def main() -> None:
    logger.info("Running clubs holdout under both decision rules")
    by_rule: dict[str, pd.DataFrame] = {
        rule: predict_holdout("club", decision_rule=rule) for rule in _RULES
    }

    # Sanity: λ and outcome predictions must be identical across rules.
    a, b = by_rule["argmax_v0"], by_rule["outcome_conditional_v0"]
    if not (a["lambda_home"].equals(b["lambda_home"]) and a["lambda_away"].equals(b["lambda_away"])):
        raise RuntimeError("λ values diverged between rules — bug")
    if not a["predicted_outcome"].equals(b["predicted_outcome"]):
        raise RuntimeError("predicted_outcome diverged between rules — bug")

    overall = {rule: _scoreline_summary(by_rule[rule]) for rule in _RULES}
    _print_block("All clubs (PL + La Liga + Ligue 1)", overall)

    for league_id, league_name in _LEAGUES.items():
        per_league = {
            rule: _scoreline_summary(by_rule[rule][by_rule[rule]["league_id"] == league_id])
            for rule in _RULES
        }
        _print_block(league_name, per_league)

    _draw_analysis(by_rule["argmax_v0"])


def _draw_analysis(df: pd.DataFrame) -> None:
    """Why don't 0-0 / 2-2 / 3-3 ever get served?

    Three things to show:
      1. How often is each diagonal cell the joint-argmax (current production)?
      2. How often is each diagonal cell the diagonal-argmax (what
         outcome_conditional would serve IF draw won the marginal)?
      3. How often does draw actually win the marginal argmax?
      4. What are actuals for 0-0 / 1-1 / 2-2 / 3-3 in this holdout?
    """
    print("\n=== Draw analysis (clubs holdout) ===")

    # Actual draw scorelines in the holdout
    actuals = Counter(zip(df["actual_home_goals"], df["actual_away_goals"], strict=True))
    n = len(df)
    draw_actuals = {(k, k): actuals.get((k, k), 0) for k in range(6)}
    total_draws = sum(draw_actuals.values())
    print(f"\nActual draw breakdown (n={n}):")
    for (h, a), c in draw_actuals.items():
        if c > 0:
            print(f"  {h}-{a}: {c:>3}  ({c / n:>5.1%})")
    print(f"  total draws: {total_draws}  ({total_draws / n:.1%})")

    # Recompute matrices to inspect joint vs diagonal argmax
    _, _, _, rho, _ = _load_artefacts("artefacts/club")
    print(f"\nClub model rho = {rho:+.4f}")

    joint_diag = Counter()      # joint-argmax-on-diagonal-cell
    diag_argmax = Counter()     # diagonal-argmax cell (per fixture)
    p_draw_max = 0.0
    p_draw_at_argmax_count = 0
    for _, row in df.iterrows():
        mat = _bivariate_poisson_matrix(row["lambda_home"], row["lambda_away"], rho)
        ji, jj = np.unravel_index(mat.argmax(), mat.shape)
        if ji == jj:
            joint_diag[(int(ji), int(jj))] += 1
        diag = np.diag(mat)
        di = int(diag.argmax())
        diag_argmax[(di, di)] += 1
        # Outcome marginals
        ph = float(np.tril(mat, -1).sum())
        pd_ = float(np.trace(mat))
        pa = float(np.triu(mat, 1).sum())
        p_draw_max = max(p_draw_max, pd_)
        if pd_ >= max(ph, pa):
            p_draw_at_argmax_count += 1

    print(
        f"\nFixtures where joint argmax IS on the diagonal: "
        f"{sum(joint_diag.values())} / {n} ({sum(joint_diag.values()) / n:.1%})",
    )
    for cell, c in sorted(joint_diag.items()):
        print(f"  {cell[0]}-{cell[1]}: {c}")

    print(
        f"\nFixtures where draw IS the marginal argmax: "
        f"{p_draw_at_argmax_count} / {n} ({p_draw_at_argmax_count / n:.1%})",
    )
    print(f"Max P(draw) across all fixtures: {p_draw_max:.3f}")

    print(
        "\nIf we *forced* a draw, which diagonal cell would we serve? "
        "(distribution of diagonal-argmax across fixtures)",
    )
    for cell, c in sorted(diag_argmax.items()):
        print(f"  {cell[0]}-{cell[1]}: {c:>4}  ({c / n:>5.1%})")

    print(
        "\nlambda_home range:  "
        f"min={df['lambda_home'].min():.2f}  "
        f"mean={df['lambda_home'].mean():.2f}  "
        f"max={df['lambda_home'].max():.2f}",
    )
    print(
        "lambda_away range:  "
        f"min={df['lambda_away'].min():.2f}  "
        f"mean={df['lambda_away'].mean():.2f}  "
        f"max={df['lambda_away'].max():.2f}",
    )


if __name__ == "__main__":
    main()
