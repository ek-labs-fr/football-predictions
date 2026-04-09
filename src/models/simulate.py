"""Monte Carlo tournament simulation (Step 3.10).

Simulates group stages with FIFA tiebreakers and knockout brackets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAX_GOALS = 8
_ET_LAMBDA_FACTOR = 1 / 3  # Extra time is ~30 min = 1/3 of 90 min
_PENALTY_HOME_WIN_PROB = 0.5


# ------------------------------------------------------------------
# Match simulation
# ------------------------------------------------------------------


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Sample a single scoreline from Poisson distributions.

    Parameters
    ----------
    lambda_home, lambda_away:
        Expected goals for each team.
    rho:
        Bivariate Poisson correlation (currently uses independent sampling).
    rng:
        Random number generator.

    Returns
    -------
    (home_goals, away_goals)
    """
    if rng is None:
        rng = np.random.default_rng()
    home_goals = rng.poisson(lambda_home)
    away_goals = rng.poisson(lambda_away)
    return int(home_goals), int(away_goals)


def simulate_knockout_match(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, str]:
    """Simulate a knockout match including extra time and penalties.

    Returns
    -------
    (home_goals, away_goals, decided_in)
        decided_in: "FT", "AET", or "PEN"
    """
    if rng is None:
        rng = np.random.default_rng()

    hg, ag = simulate_match(lambda_home, lambda_away, rho, rng)
    if hg != ag:
        return hg, ag, "FT"

    # Extra time (inflated λ by 1/3)
    et_h, et_a = simulate_match(
        lambda_home * _ET_LAMBDA_FACTOR, lambda_away * _ET_LAMBDA_FACTOR, rho, rng
    )
    hg += et_h
    ag += et_a
    if hg != ag:
        return hg, ag, "AET"

    # Penalty shootout (50/50)
    if rng.random() < _PENALTY_HOME_WIN_PROB:
        return hg + 1, ag, "PEN"
    return hg, ag + 1, "PEN"


# ------------------------------------------------------------------
# Group stage simulation
# ------------------------------------------------------------------


@dataclass
class GroupResult:
    """Per-team group stage result for one simulation."""

    team_id: int
    points: int = 0
    goal_diff: int = 0
    goals_scored: int = 0
    wins: int = 0


def simulate_group_stage(
    group_teams: list[int],
    get_lambdas: callable,  # type: ignore[valid-type]
    n_sims: int = 10000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate a group stage and return advancement probabilities.

    Parameters
    ----------
    group_teams:
        List of team IDs in the group.
    get_lambdas:
        Callable(home_id, away_id) -> (lambda_home, lambda_away).
    n_sims:
        Number of Monte Carlo simulations.
    rng:
        Random number generator.

    Returns
    -------
    pd.DataFrame with columns: team_id, 1st_prob, 2nd_prob, 3rd_prob, advance_prob
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_teams = len(group_teams)
    finish_counts: dict[int, dict[int, int]] = {
        t: {pos: 0 for pos in range(n_teams)} for t in group_teams
    }

    # Pre-compute lambdas for all matchups
    matchups: list[tuple[int, int, float, float]] = []
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            lh, la = get_lambdas(group_teams[i], group_teams[j])
            matchups.append((group_teams[i], group_teams[j], lh, la))

    for _ in range(n_sims):
        results: dict[int, GroupResult] = {t: GroupResult(team_id=t) for t in group_teams}

        for home_id, away_id, lh, la in matchups:
            hg, ag = simulate_match(lh, la, rng=rng)
            rh, ra = results[home_id], results[away_id]

            rh.goals_scored += hg
            rh.goal_diff += hg - ag
            ra.goals_scored += ag
            ra.goal_diff += ag - hg

            if hg > ag:
                rh.points += 3
                rh.wins += 1
            elif ag > hg:
                ra.points += 3
                ra.wins += 1
            else:
                rh.points += 1
                ra.points += 1

        # Sort by FIFA tiebreakers: points → GD → GS
        ranking = sorted(
            results.values(),
            key=lambda r: (r.points, r.goal_diff, r.goals_scored),
            reverse=True,
        )
        for pos, gr in enumerate(ranking):
            finish_counts[gr.team_id][pos] += 1

    rows = []
    for team_id, counts in finish_counts.items():
        row: dict[str, Any] = {"team_id": team_id}
        for pos in range(n_teams):
            row[f"pos_{pos + 1}_prob"] = counts[pos] / n_sims
        row["advance_prob"] = (counts[0] + counts[1]) / n_sims  # top 2 advance
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Full tournament simulation
# ------------------------------------------------------------------


def simulate_tournament(
    groups: dict[str, list[int]],
    get_lambdas: callable,  # type: ignore[valid-type]
    n_sims: int = 10000,
    rho: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate a full tournament: group stage → knockout → champion.

    Parameters
    ----------
    groups:
        Dict mapping group name to list of team IDs, e.g. {"A": [1,2,3,4]}.
    get_lambdas:
        Callable(home_id, away_id) -> (lambda_home, lambda_away).
    n_sims:
        Number of Monte Carlo simulations.
    rho:
        Bivariate Poisson correlation parameter.
    rng:
        Random number generator.

    Returns
    -------
    pd.DataFrame with columns: team, group_win_prob, advance_prob, qf_prob, sf_prob,
        final_prob, champion_prob.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_teams = [t for teams in groups.values() for t in teams]
    stage_counts: dict[int, dict[str, int]] = {
        t: {"group_win": 0, "advance": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for t in all_teams
    }

    for _sim in range(n_sims):
        # Group stage
        group_winners: dict[str, list[int]] = {}
        for gname, gteams in groups.items():
            results: dict[int, GroupResult] = {t: GroupResult(team_id=t) for t in gteams}

            for i in range(len(gteams)):
                for j in range(i + 1, len(gteams)):
                    lh, la = get_lambdas(gteams[i], gteams[j])
                    hg, ag = simulate_match(lh, la, rng=rng)
                    rh, ra = results[gteams[i]], results[gteams[j]]
                    rh.goals_scored += hg
                    rh.goal_diff += hg - ag
                    ra.goals_scored += ag
                    ra.goal_diff += ag - hg
                    if hg > ag:
                        rh.points += 3
                    elif ag > hg:
                        ra.points += 3
                    else:
                        rh.points += 1
                        ra.points += 1

            ranking = sorted(
                results.values(),
                key=lambda r: (r.points, r.goal_diff, r.goals_scored),
                reverse=True,
            )
            group_winners[gname] = [ranking[0].team_id, ranking[1].team_id]
            stage_counts[ranking[0].team_id]["group_win"] += 1
            for r in ranking[:2]:
                stage_counts[r.team_id]["advance"] += 1

        # Build bracket (simplified: 1A vs 2B, 1B vs 2A, etc.)
        group_names = sorted(groups.keys())
        r16_teams = []
        for i in range(0, len(group_names), 2):
            if i + 1 < len(group_names):
                ga, gb = group_names[i], group_names[i + 1]
                r16_teams.append((group_winners[ga][0], group_winners[gb][1]))
                r16_teams.append((group_winners[gb][0], group_winners[ga][1]))
            else:
                gn = group_names[i]
                r16_teams.append((group_winners[gn][0], group_winners[gn][1]))

        # Knockout rounds
        def _run_round(matchups: list[tuple[int, int]], stage_name: str) -> list[int]:
            winners = []
            for t1, t2 in matchups:
                lh, la = get_lambdas(t1, t2)
                hg, ag, _ = simulate_knockout_match(lh, la, rho, rng)
                winner = t1 if hg > ag else t2
                winners.append(winner)
                stage_counts[winner][stage_name] += 1
            return winners

        qf_teams = _run_round(r16_teams, "qf")
        sf_matchups = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, len(qf_teams), 2)]
        sf_teams = _run_round(sf_matchups, "sf")
        final_matchups = [(sf_teams[i], sf_teams[i + 1]) for i in range(0, len(sf_teams), 2)]
        final_teams = _run_round(final_matchups, "final")

        if final_teams:
            # Final
            if len(final_teams) >= 2:
                lh, la = get_lambdas(final_teams[0], final_teams[1])
                hg, ag, _ = simulate_knockout_match(lh, la, rho, rng)
                champion = final_teams[0] if hg > ag else final_teams[1]
            else:
                champion = final_teams[0]
            stage_counts[champion]["champion"] += 1

    rows = []
    for team_id, counts in stage_counts.items():
        rows.append(
            {
                "team_id": team_id,
                "group_win_prob": counts["group_win"] / n_sims,
                "advance_prob": counts["advance"] / n_sims,
                "qf_prob": counts["qf"] / n_sims,
                "sf_prob": counts["sf"] / n_sims,
                "final_prob": counts["final"] / n_sims,
                "champion_prob": counts["champion"] / n_sims,
            }
        )

    return pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)
