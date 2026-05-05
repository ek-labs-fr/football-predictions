"""One-line plain-English rationale per fixture.

For each fixture, the Poisson Linear model gives us per-feature coefficients in
both the home-goals and away-goals regressors. The signed contribution of
feature i to the home-vs-away decision is::

    contribution_i = (coef_home_i - coef_away_i) * scaled_x_i

We pick the largest |contribution| as the "top driver" and look up a phrase
template keyed by feature name + direction. Templates use plain English (no
"xG", no Greek letters) and substitute team names plus the raw (unscaled)
feature value.

Used by ``src.inference.predict`` to attach a ``rationale`` field to every
upcoming/recent/holdout match payload. Recomputed each run; not persisted in
the frozen prediction store, so retrains automatically refresh the text.
"""

from __future__ import annotations

import numpy as np

# Each entry: (phrase_when_pulling_toward_home, phrase_when_pulling_toward_away)
# Templates accept {home}, {away}, {value} (raw, unscaled feature value).
#
# Direction convention: signed_contribution > 0 ⇒ feature pulls the prediction
# toward the home side ⇒ first phrase is rendered.
_PHRASES: dict[str, tuple[str, str]] = {
    # ---- Recent chance creation / concession (xG) ----
    "home_xg_for_avg_l10": (
        "{home} have been creating {value:.1f} scoring chances per match recently.",
        "{home} have struggled to create chances ({value:.1f} per match).",
    ),
    "away_xg_for_avg_l10": (
        "{away} have looked toothless going forward ({value:.1f} chances/match).",
        "{away} have been creating {value:.1f} chances/match — even on the road.",
    ),
    "home_xg_against_avg_l10": (
        "{home} have looked tight at the back recently.",
        "{home} have been giving up {value:.1f} scoring chances per match.",
    ),
    "away_xg_against_avg_l10": (
        "{away} have been letting in {value:.1f} chances/match — defensive holes.",
        "{away} have been a tough nut to crack defensively.",
    ),
    # ---- Squad quality ----
    "squad_rating_diff": (
        "{home} hold a clear squad-quality edge.",
        "{away} are rated stronger on paper.",
    ),
    "home_squad_avg_rating": (
        "{home} field one of the strongest squads around.",
        "{home}'s squad rating has been below par.",
    ),
    "away_squad_avg_rating": (
        "{away} are short on top-tier players.",
        "{away} field one of the strongest squads — even on the road.",
    ),
    "home_squad_goals_club_season": (
        "{home}'s squad has been prolific this season.",
        "{home}'s squad has been quiet on the scoring front this season.",
    ),
    "away_squad_goals_club_season": (
        "{away}'s players have gone cold in front of goal this season.",
        "{away}'s players have been finding the net all season.",
    ),
    "home_top5_league_ratio": (
        "{home}'s squad pedigree is a clear edge.",
        "{home}'s squad pedigree is below the league norm.",
    ),
    "away_top5_league_ratio": (
        "{away}'s squad pedigree is below the league norm.",
        "{away}'s squad pedigree is a clear edge.",
    ),
    # ---- Recent form ----
    "form_diff": (
        "{home} are the more in-form side.",
        "{away} have been the more in-form side.",
    ),
    "home_points_per_game_l10": (
        "{home} are riding a strong run of form.",
        "{home} have been on a poor run of results.",
    ),
    "away_points_per_game_l10": (
        "{away} are on a poor run of results.",
        "{away} arrive in good form.",
    ),
    "home_win_rate_l10": (
        "{home} have been winning regularly.",
        "{home} have rarely won lately.",
    ),
    "away_win_rate_l10": (
        "{away} have rarely won lately.",
        "{away} have been winning regularly.",
    ),
    # ---- Recent goal-scoring (raw goals) ----
    "home_goals_scored_avg_l10": (
        "{home} have averaged {value:.1f} goals per match recently.",
        "{home} have only managed {value:.1f} goals per match lately.",
    ),
    "away_goals_scored_avg_l10": (
        "{away} have struggled to score recently ({value:.1f}/match).",
        "{away} have been scoring freely ({value:.1f}/match).",
    ),
    "goals_scored_avg_diff": (
        "{home} have been the sharper side in front of goal lately.",
        "{away} have been the sharper side in front of goal lately.",
    ),
    # ---- Recent goal-conceding (raw goals) ----
    "home_goals_conceded_avg_l10": (
        "{home} have been tight at the back.",
        "{home} have been leaking {value:.1f} goals per match.",
    ),
    "away_goals_conceded_avg_l10": (
        "{away} have been shipping {value:.1f} goals per match.",
        "{away} have looked defensively solid.",
    ),
    "home_clean_sheet_rate_l10": (
        "{home} have been keeping plenty of clean sheets.",
        "{home} have rarely kept a clean sheet.",
    ),
    "away_clean_sheet_rate_l10": (
        "{away} have rarely kept a clean sheet.",
        "{away} have been keeping plenty of clean sheets.",
    ),
    # ---- Head-to-head ----
    "h2h_home_goals_avg": (
        "{home} have historically dominated this fixture.",
        "{home} have struggled in past meetings with {away}.",
    ),
    "h2h_away_goals_avg": (
        "{away} have struggled in past meetings with {home}.",
        "{away} have a strong record against {home}.",
    ),
    "h2h_home_win_rate": (
        "{home} usually win this fixture.",
        "{away} usually have the upper hand in this fixture.",
    ),
    # ---- Schedule ----
    "rest_days_diff": (
        "{home} have a rest-day advantage.",
        "{away} arrive better-rested.",
    ),
    "home_rest_days": (
        "{home} have had plenty of recovery time.",
        "{home} are coming off a short rest.",
    ),
    "away_rest_days": (
        "{away} are coming off a short rest.",
        "{away} have had plenty of recovery time.",
    ),
}


_FALLBACK_HOME = "Squad and form indicators tilt this one toward {home}."
_FALLBACK_AWAY = "Squad and form indicators tilt this one toward {away}."
_DRAW_TEMPLATE = "{home} and {away} look evenly matched on form and squad."


def render_rationale(
    home_team: str,
    away_team: str,
    predicted_outcome: str,
    feature_cols: list[str],
    scaled_x: np.ndarray,
    raw_x: np.ndarray,
    coef_home: np.ndarray,
    coef_away: np.ndarray,
) -> str:
    """Pick a feature in the direction of the prediction and explain it.

    All vector inputs are 1-D arrays of length ``len(feature_cols)``.
    ``scaled_x`` is the StandardScaler-transformed feature row; ``raw_x`` is
    the original feature row before scaling, used for human-readable values.

    For home/away win predictions we pick the strongest contributor pulling
    in the predicted direction — this guarantees the rationale explains why
    the predicted side was favoured, rather than highlighting a feature
    pulling the opposite way that other features overcame. For draws the
    contributors don't favour either side meaningfully, so we use a generic
    "evenly matched" template.
    """
    contribs = (coef_home - coef_away) * scaled_x

    if predicted_outcome == "home_win":
        idx = int(np.argmax(contribs))
        if contribs[idx] <= 0:
            return _FALLBACK_HOME.format(home=home_team, away=away_team)
        toward_home = True
    elif predicted_outcome == "away_win":
        idx = int(np.argmin(contribs))
        if contribs[idx] >= 0:
            return _FALLBACK_AWAY.format(home=home_team, away=away_team)
        toward_home = False
    else:  # draw
        return _DRAW_TEMPLATE.format(home=home_team, away=away_team)

    feature = feature_cols[idx]
    raw_value = float(raw_x[idx]) if not np.isnan(raw_x[idx]) else 0.0

    if feature in _PHRASES:
        toward_home_template, toward_away_template = _PHRASES[feature]
        template = toward_home_template if toward_home else toward_away_template
    else:
        template = _FALLBACK_HOME if toward_home else _FALLBACK_AWAY

    return template.format(home=home_team, away=away_team, value=raw_value)
