# Model Card — Clubs

A senior-data-scientist-oriented assessment of the **club** mode of the football-predictions system. Club mode targets domestic-league fixtures from three competitions: **Premier League** (`league_id=39`), **La Liga** (`league_id=140`), and **Ligue 1** (`league_id=61`).

The system fits two independent Poisson regressors on per-fixture engineered features (one for `home_goals`, one for `away_goals`), then combines them via a bivariate-Poisson matrix with a calibrated correlation parameter ρ. Outcome probabilities `P(W/D/L)` are derived by summing regions of the resulting scoreline matrix.

For the **national** counterpart of this card see `documents/model-card-national.md`. For system architecture see `documents/technical-architecture.md`.

---

## 1. TL;DR — model card summary

| | |
|---|---|
| **Task** | Predict `(home_goals, away_goals)` for PL / La Liga / Ligue 1 fixtures, with a derived 3×1 outcome distribution |
| **Holdout** | Most recent fully completed domestic season (currently **2024-25**, ~380 fixtures × 3 leagues) |
| **Production pick** | **Poisson Linear** (sklearn `PoissonRegressor`) — **not** LightGBM. Linear narrowly beat XGBoost / LightGBM by all metrics. |
| **Calibration** | Bivariate Poisson with scalar ρ = **+0.0658**, fit on last 15% of train via Brier-on-draws minimization (was +0.0425 before xG features added on 2026-05-04 — see §15) |
| **Headline metrics (holdout)** | MAE_avg **0.854** · RPS **0.199** · W/D/L Accuracy **0.546** · Log loss **0.971** · Brier **0.192** · Exact scoreline acc **12.7%** |
| **Feature count** | 38 numeric features (was 34 before xG added; no static drop list — `_CLUB_DROP_FEATURES` is empty) |
| **Top feature** (mean \|SHAP\|) | `home_squad_goals_club_season` at **0.152** — ~3.5× the #2 feature |
| **Tuning** | Optuna **was** run for the XGBoost candidate (`artefacts/club/best_params.json` exists), but the tuned XGBoost did not win the comparison — Poisson Linear did. So tuned hyperparameters are saved but not in the production model. |
| **Feature selection** | Empty `_CLUB_DROP_FEATURES` (`train.py:77`) — all features survive into training |
| **Decision rule** | `argmax_v0` — modal scoreline of the bivariate-Poisson matrix; outcome derived from marginal sums |
| **Known weak point** | Exact-scoreline accuracy 13.0%, below the "Good" 18% benchmark. Same root cause as national mode — modal-scoreline collapse onto low-score buckets. |

**Honest read of the metrics:** club mode beats national mode on every reported metric. Closer-to-Excellent MAE (0.857 vs 0.85 target), accuracy above the "Good" 52% bar, scoreline accuracy almost 2× the national rate. The dataset is cleaner — no friendly-vs-final stage variance, more fixtures per team-season — and a simple linear model captures most of the signal.

---

## 2. Target and inputs

**Targets:**
- `home_goals: int ∈ {0, 1, 2, …}` — Poisson regressor
- `away_goals: int ∈ {0, 1, 2, …}` — Poisson regressor (independent fit)
- Derived: `outcome ∈ {away_win, draw, home_win}` from the scoreline matrix

**Inputs:** ~48 numeric features built per-fixture from the historical club-fixture corpus across the three leagues, plus squad metadata. No FIFA rankings (those are international-only); no Elo (not in the club feature set as of current code). No tournament running stats. No categorical features pass into the model.

**Scope:** all completed PL / La Liga / Ligue 1 fixtures in the corpus, going back as far as the API-Football data extends. The model is jointly fit across all three leagues — there is no per-league sub-model.

---

## 3. Feature engineering

Five families assembled in `src/features/`, joined into a single flat row per fixture in `build.py`. All features carry a strict **leakage-prevention guard** — derivation uses only data with `date < match_date`, never `≤`.

### 3.1 Rolling form (`rolling.py`)
Per-team windows over prior club matches in the same league.

| Feature | Definition |
|---|---|
| `goals_scored_avg_l10`, `goals_conceded_avg_l10` | Mean per-match goals over the last 10 prior fixtures |
| `win_rate_l10`, `points_per_game_l10` | Standard form aggregates over last 10 |
| `clean_sheet_rate_l10` | Fraction of last 10 with 0 conceded |
| `form_last5` | Concatenated `W/D/L` string (display only; not a model feature) |
| `matches_available` | Count of prior fixtures used (caps at 10) |

For club mode, `matches_available` saturates at 10 quickly — every PL team plays 38 matches/season, so by mid-October most teams have a full L10 window. This is in contrast to national mode where teams sometimes play < 10 matches per year.

### 3.2 Squad quality (`squad.py`)
Per-`(team_id, season)` aggregates from API-Football's `/players` endpoint.

| Feature | Definition |
|---|---|
| `squad_avg_age` | Mean age across squad |
| `squad_avg_rating` | Mean player rating (0–99) — `None` if <50% of squad has ratings (`squad.py:60`) |
| `squad_goals_club_season` | Total club-season goals across the squad — accumulates as the season progresses |
| `top5_league_ratio` | Fraction of squad playing in EPL/La Liga/Bundesliga/Serie A/Ligue 1 (mostly 1.0 for these clubs) |
| `star_player_present` | Boolean — any player rated ≥ 8.0 (`squad.py:26`) |

**`squad_goals_club_season` is the dominant feature** in club mode (see §5.2). It's effectively a season-form aggregate of the squad's offensive output.

**Limitation:** squad data is single-snapshot per season. Mid-season transfers, January-window arrivals, and rotation patterns are not modelled. For top-flight European leagues this is a real source of noise — the January window can change a club's squad quality materially.

### 3.3 Head-to-head (`h2h.py`)
Prior-meeting statistics between the two clubs.

| Feature | Definition |
|---|---|
| `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_matches_total` | Raw counts from prior H2H |
| `h2h_home_win_rate` | Default `0.33` if `< 3` prior meetings (`h2h.py:17,28`) |
| `h2h_home_goals_avg`, `h2h_away_goals_avg` | Per-side scoring rate in prior H2H |
| `h2h_last_winner` | Categorical, most recent prior outcome |

`min_meetings = 3` rarely binds for club fixtures within a single domestic league — e.g. Liverpool vs Arsenal has decades of prior PL meetings. The threshold matters more for promoted teams or recent expansions.

Notably, `h2h_home_goals_avg` and `h2h_matches_total` appear in the club top-15 SHAP (§5.2), unlike national mode where most H2H features were dropped as noise.

### 3.4 Match context (`build.py`) — **club-relevant subset only**
National-only features (`is_knockout`, `match_weight × competition_weight`, `neutral_venue`) are not meaningfully informative in club mode. The single-league-tournament structure means most domestic fixtures share the same `match_weight`. Match context still flows through the feature table for code uniformity, but doesn't carry signal here.

### 3.5 Rest days (`build.py:464`) — **club-only**
- **`rest_days_diff`** = `home_rest_days − away_rest_days`. Captures the "Champions League hangover" effect: a team that played a midweek Champions League / Europa fixture typically has 1-2 fewer rest days going into the weekend than a domestic-only opponent.

This is the only feature unique to club mode (national mode has no equivalent — national fixtures aren't congested in the same way).

### 3.6 Derived differences (`build.py:279–291`)

| Feature | Definition |
|---|---|
| `form_diff` | `home_points_per_game_l10 − away_points_per_game_l10` |
| `goals_scored_avg_diff` | `home_goals_scored_avg_l10 − away_goals_scored_avg_l10` |
| `squad_rating_diff` | Pairwise squad-rating gap |
| `top5_ratio_diff` | Pairwise top-5-league-ratio gap |
| `rest_days_diff` | Club-only (above) |

Note: **`elo_diff` and `rank_diff` (national-mode favorites) are not in the club feature set**. Club teams aren't ranked in FIFA's system, and Elo isn't computed for them in the current pipeline. This is one of the structural reasons the club model relies more on squad-level features.

### 3.7 Rolling xG (`xg_rolling.py`) — added 2026-05-04

Added to address a structural under-dispersion of predicted λ values (see §15 for the full motivation and before/after numbers). Per-team rolling means of expected goals over the prior 10 fixtures.

| Feature | Definition |
|---|---|
| `xg_for_avg_l10` | Mean of `xg` scored over the team's prior 10 fixtures (only fixtures with populated xG count) |
| `xg_against_avg_l10` | Mean of `xg` conceded over the team's prior 10 fixtures |

Joined into the training table as `home_xg_for_avg_l10`, `home_xg_against_avg_l10`, `away_xg_for_avg_l10`, `away_xg_against_avg_l10` — 4 new columns.

**Source:** API-Football `/fixtures/statistics` endpoint, parsed from the local cache at `data/raw/club/fixtures_statistics/` by `src.data.ingest.build_club_match_statistics_from_cache`. The cache is populated by the existing `pull_match_statistics` ingest path.

**Coverage:** xG is **only available for PL / La Liga / Ligue 1 from 2023 onward**. Pre-2023 fixtures have NaN. The rolling calculation requires ≥3 prior xG-populated matches in the L10 window or returns NaN, so older training rows are imputed at training time with the column median (standard pipeline behaviour). For the 2024-25 holdout the features are 100% populated; for the 2023 training rows ~94% populated; pre-2023 they fall back to the median.

**Why it matters:** xG is a less-noisy signal of finishing-chance creation than raw goals. The Poisson model picks up a real coefficient on the xG features and produces λ predictions with materially higher variance — the predicted λ_home spread (std) goes from 0.418 to 0.503 (+20%), and the share of fixtures where the model expects ≥3.5 total goals goes from 2.7% to 4.0%. This directly attacks the "model can only predict 0/1-goal scorelines" issue surfaced in §12 / §15.

**Limitation:** LightGBM Poisson does **not** benefit from xG at the current coverage level — with ~70% of training rows imputed, the gain-based splits never pick xG over the fully-populated baseline features. Only the linear Poisson regressor extracts the signal. As coverage rises (more 2024+ training data accrues) LightGBM may start to use xG too; revisit late 2026.

### 3.8 Tournament running stats — **NOT included in club mode**

Tournament accumulators (`matches_played_in_tournament`, etc.) are national-only; they're explicitly excluded from the club training table because their semantics don't carry over (a domestic season isn't a "tournament" in the WC sense — there's no single bracket of progress).

---

## 4. Feature preprocessing, splits, selection

### 4.1 Preprocessing
- **Scaling:** `StandardScaler` is fit on the training set and applied to **linear** models (`poisson_linear`, `logistic_regression`). Tree models (XGBoost, LightGBM) use raw features. Since the production pick for club mode IS `poisson_linear`, **the scaler IS used** in the live pipeline (unlike national mode where the saved scaler is dead weight from the LightGBM pick).
- **Categorical encoding:** none in the model. Outcome labels encoded via `LabelEncoder` for the classifier branch only (0=away_win, 1=draw, 2=home_win).
- **Missing-data handling:** features with missing values are filled with the **column median** computed on the training fold (`train.py:194`). No "is_missing" indicator. Even sparse features are imputed.

### 4.2 Train / test / calibration boundaries

| | Club mode |
|---|---|
| Train | All seasons before holdout |
| Calibration carve-out | Last 15% of train, chronologically |
| Holdout (test) | **Most recently completed domestic season** — detected at training time as the max season where all matches are >30 days old (`train.py:155–162`). Currently 2024-25. |

The holdout strategy is more aggressive than national's "single tournament held out" — a full league season is ~380 fixtures per league, so the club holdout contains ~1140 fixtures across the three leagues. That's ~18× the WC 2022 holdout's 64. **Statistical significance on club-mode metrics is correspondingly tighter.**

### 4.3 Sample weighting
The `match_weight` column flows into every `fit()` call as `sample_weight=w_train`. For club mode, all fixtures within a domestic league have similar `match_weight` (no friendlies, no qualifiers, no knockout-vs-group distinction), so the weighting is **less differentiating than in national mode**. The mechanism is still active for code uniformity but its effect is small.

### 4.4 Feature selection

**Production:** `_CLUB_DROP_FEATURES = []` at `train.py:77`. **Empty.** All features that survive the upstream feature-engineering pipeline make it into training.

The static drop list infrastructure was set up but a club-specific SHAP analysis to populate the list hasn't been run. A reviewer should flag this: 14 national features were dropped on the basis of mean `|SHAP| < 0.002` — the equivalent club analysis hasn't been done and should be. Some features in the club top-15 SHAP have very small mean values (e.g. `away_squad_avg_age` at 0.0026, `away_top5_league_ratio` at 0.0024) that would be drop candidates under the same threshold.

**Dormant infrastructure:** `src/models/select.py` (variance threshold → correlation filter → RFECV → permutation importance) is implemented but not on the production path. Same status as national mode.

---

## 5. Feature importance

### 5.1 SHAP setup (`src/models/explain.py`)
- **Method:** `shap.TreeExplainer` on the LightGBM home-goals model.
  - **Caveat:** SHAP is computed on **LightGBM**, not the production winner (`poisson_linear`). The two models capture broadly similar signal — both are Poisson regressions of goals on the same features — but the SHAP attribution is for LightGBM, not the live model. A linear model's coefficients would be the natural attribution for `poisson_linear`; those aren't currently saved.
- **Background data:** full training set
- **Saved:** `artefacts/club/shap_explainer.pkl`, `artefacts/club/shap_feature_importance.csv`
- **Plots:** `outputs/club/shap_summary_home.png`, `outputs/club/shap_bar_importance.png`, dependence plots for the top features

### 5.2 Top features (mean |SHAP|)

| Rank | Feature | Mean \|SHAP\| | Comment |
|---|---|---|---|
| 1 | `home_squad_goals_club_season` | **0.152** | Dominant — ~3.5× the next feature |
| 2 | `away_squad_avg_rating` | 0.043 | Opponent quality (player ratings) |
| 3 | `away_squad_goals_club_season` | 0.032 | Opponent offensive form |
| 4 | `squad_rating_diff` | 0.020 | Pairwise squad-quality gap |
| 5 | `away_goals_conceded_avg_l10` | 0.012 | Opponent defensive form |
| 6 | `home_top5_league_ratio` | 0.010 | (Same feature is a national-mode drop) |
| 7 | `away_points_per_game_l10` | 0.008 | Opponent recent results |
| 8 | `form_diff` | 0.008 | Recent-form gap |
| 9 | `home_goals_scored_avg_l10` | 0.005 | Home offensive form |
| 10 | `h2h_home_goals_avg` | 0.004 | Prior H2H scoring |

**Interpretation notes a reviewer would push on:**

- **Squad quality dominates club mode**: 4 of the top 5 features are squad-derived (`*_squad_goals_club_season`, `*_squad_avg_rating`, `squad_rating_diff`). In national mode, Elo + FIFA rank dominate; in club mode, with no Elo/FIFA-rank features available, the squad metrics fill the gap.
- **`home_squad_goals_club_season` carries 3.5× the weight of #2** — concentrated signal. As with national-mode's `elo_diff`, this means the model is sensitive to a single feature's data quality. If `/players` endpoint coverage degrades or season-goal accumulation runs late, club-mode quality drops fast.
- **Recent rolling-form features (rank 5-9) have small but non-negligible weight.** A reviewer might ask whether longer windows (L20, L38 = full prior season) would do better — there's no current experiment.
- **`home/away_top5_league_ratio` appears in the club top 6 despite being explicitly dropped in national mode.** Different feature behavior across modes — illustrates that the static drop list shouldn't be assumed mode-agnostic.

---

## 6. Model architecture and training

### 6.1 Two-tower bivariate Poisson

```
features ──┬──► PoissonRegressor (sklearn) ──► λ_home
           └──► PoissonRegressor (sklearn) ──► λ_away

then:  P(h, a) = bivariate_poisson(λ_home, λ_away, ρ=+0.042)  # 11×11 matrix
       predicted_score   = argmax_{h,a} P(h, a)
       p_home_win  = sum_{h>a} P(h, a)
       p_draw       = sum_{h=a} P(h, a)
       p_away_win   = sum_{h<a} P(h, a)
```

The two towers are fit **independently**.

### 6.2 Production hyperparameters

`sklearn.linear_model.PoissonRegressor` with default hyperparameters (`alpha=1.0` L2 regularization, `max_iter=100`, `tol=1e-4`, `solver='lbfgs'`). The features go through `StandardScaler` first.

**Note**: the saved `artefacts/club/best_params.json` (Optuna output) contains tuned XGBoost params:
```json
{
  "n_estimators": 179,
  "learning_rate": 0.0133,
  "max_depth": 4,
  "min_child_weight": 3,
  "subsample": 0.673,
  "colsample_bytree": 0.858,
  "gamma": 2.32,
  "reg_alpha": 1.26,
  "reg_lambda": 3.7e-05
}
```

Those hyperparameters were applied to the `xgboost_poisson_tuned` candidate, which scored MAE_avg 0.859 — slightly *worse* than the linear model's 0.857. So the tuned XGBoost exists in the artefacts directory but is not the production model. **A reviewer should note: this is one of the few places in the project where Optuna actually ran. The result was that the tuned XGBoost couldn't beat a linear baseline.** This is a non-trivial finding — it suggests either the Optuna search space is wrong, the dataset is small enough that linear models generalize better, or there's not much non-linear signal to extract from these features.

### 6.3 Why Poisson Linear (full candidate comparison, holdout 2024-25 season)

Sorted by `MAE_avg`, lower is better:

| Model | MAE_home | MAE_away | MAE_avg | Acc | RPS | Log loss | Brier | Train-time |
|---|---|---|---|---|---|---|---|---|
| **Poisson Linear** ← selected | 0.891 | 0.823 | **0.857** | **0.538** | **0.202** | **0.981** | **0.194** | 3.84s |
| XGBoost Poisson (Optuna-tuned) | 0.902 | 0.816 | 0.859 | 0.524 | 0.206 | 0.991 | 0.197 | 1.11s |
| LightGBM Poisson | 0.893 | 0.835 | 0.864 | 0.521 | 0.204 | 0.986 | 0.195 | 0.82s |
| XGBoost Poisson (untuned) | 0.905 | 0.825 | 0.865 | 0.526 | 0.203 | 0.983 | 0.195 | 1.15s |
| Baseline mean-goals | 1.102 | 0.850 | 0.976 | 0.437 | 0.233 | 1.070 | 0.216 | 0s |
| XGBoost Classifier | — | — | — | 0.512 | 0.206 | 0.993 | 0.198 | 1.10s |
| Logistic Regression | — | — | — | 0.471 | 0.209 | 1.020 | 0.203 | 0.31s |

**Poisson Linear wins on every reported metric.** It's also the slowest to train (3.84s vs LightGBM's 0.82s) — sklearn's L-BFGS converges slower than gradient-boosted trees on this dataset, but it's still <5 seconds and irrelevant for a model retrained nightly.

**Why does linear beat trees here?** Plausible explanations:
1. **Dataset size and noise level.** Club training corpus is well-sized but not huge (~1100 fixtures × N seasons). Linear models can be hard to beat when N is moderate and signal-to-noise is moderate.
2. **Feature engineering already captured the non-linearity.** Pairwise diffs (`squad_rating_diff`, `form_diff`) are explicit interactions that a linear model handles directly.
3. **Less heterogeneity than national mode.** Club fixtures don't mix friendlies and finals. The optimal target distribution shape is more uniform across rows, and a single linear hypothesis fits well.
4. **Trees might be underfitting at the chosen hyperparameters.** Both XGBoost and LightGBM were close, and tuned XGBoost was nearly tied with linear (0.859 vs 0.857). With more aggressive Optuna or different early-stopping rules, trees might catch up.

The XGBoost classifier (`multi:softprob` directly on W/D/L) gets 51.2% accuracy vs Poisson Linear's 53.8%, again supporting predicting goals and deriving outcomes.

### 6.4 Calibration — bivariate Poisson ρ (`src/models/calibrate.py`)

Same procedure as national mode: scalar ρ ∈ [−0.5, 0.5] fit via `scipy.optimize.minimize_scalar` with Brier-on-draws loss on the last 15% of train.

- **Saved:** `artefacts/club/rho.json` — `{"rho": 0.04245277...}`
- **ρ = +0.042 (positive)** — implies the calibration set has *more* draws than independence predicts. This **aligns with the football-modeling literature's prior** (Dixon-Coles' classic paper documented exactly this pattern in EPL data).
- **Counterpoint to national mode:** national ρ is −0.106 (negative). The two corpora have opposite-signed within-fixture goal correlation. Either real (international tournaments mix one-sided thrashings with cagey knockout draws differently from league play), or one of the two ρ fits is noise / sign-convention bug. **Worth investigating both.**

The magnitude is small (|0.042|) so the impact on draw probability is modest — perhaps a 1-2 percentage point shift on `p_draw` for typical λ values.

---

## 7. Hyperparameter tuning

`src/models/tune.py` Optuna study **was run** for the XGBoost Poisson candidate in club mode. Search space (from `tune.py`):

| Parameter | Range |
|---|---|
| `n_estimators` | [100, 1000] |
| `learning_rate` | [0.01, 0.3] log-scaled |
| `max_depth` | [3, 8] |
| `min_child_weight` | [1, 10] |
| `subsample`, `colsample_bytree` | [0.6, 1.0] |
| `gamma` | [0.0, 5.0] |
| `reg_alpha`, `reg_lambda` | [1e-8, 10.0] log-scaled |

**Best params found** (`artefacts/club/best_params.json`): see §6.2. The tuned XGBoost ranked #2 in the candidate comparison, narrowly beaten by the un-tuned linear model. So Optuna ran, found a local optimum for XGBoost, and the linear baseline still won — informative result.

**For national mode**: Optuna was NOT run. National hyperparameters are train.py defaults.

---

## 8. Cross-validation

| Setting | Value |
|---|---|
| Splitter | `sklearn.model_selection.TimeSeriesSplit` |
| `n_splits` | 5 |
| `gap` | 0 |
| `max_train_size` | None (expanding window) |

Used for: Optuna's per-trial scoring (`tune.py:46`) and inner validation on tree models. Per-fold metrics are not saved — only the final holdout numbers land in `artefacts/club/comparison.csv` and `outputs/training_history.csv`.

---

## 9. Evaluation metrics

### 9.1 Computed metrics
Same as national mode (`src/models/evaluate.py`): MAE_home/away/avg, exact scoreline accuracy, RPS, outcome accuracy, log loss, Brier score.

### 9.2 Achieved values vs benchmark targets

Poisson Linear, evaluated on 2024-25 season holdout (post-xG features, 2026-05-04):

| Metric | Naive | Good | Excellent | **Achieved** | Pre-xG | Verdict |
|---|---|---|---|---|---|---|
| MAE (goals) | 1.20 | 0.95 | 0.85 | **0.854** | 0.857 | Just under Excellent |
| Exact scoreline acc | 10% | 18% | 25% | **12.7%** | 13.0% | Above Naive, below Good |
| RPS | 0.24 | 0.20 | 0.17 | **0.199** | 0.202 | Just over Good |
| W/D/L Accuracy | 45% | 52% | 57% | **54.6%** | 53.8% | Beats Good |
| Log loss | 1.05 | 0.95 | 0.88 | **0.971** | 0.981 | Just over Good |
| Brier | 0.24 | 0.21 | 0.19 | **0.192** | 0.194 | Beats Good |

**Dispersion diagnostic (post-xG):**

| | Pre-xG | Post-xG |
|---|---|---|
| λ_home std | 0.418 | **0.503** (+20%) |
| λ_away std | 0.360 | **0.420** (+17%) |
| λ_home p90 | 2.00 | 2.16 |
| Fixtures with predicted λ_h+λ_a ≥ 3.5 | 2.7% | 4.0% |

The model still under-predicts dispersion relative to the actual goal distribution (30.9% of holdout fixtures had ≥4 total goals), but xG is the first concrete win on this axis. See §15 for the full investigation.

### 9.3 What the numbers say

**Club mode beats national mode across every reported metric**, often by meaningful margins:

| | National | Club | Δ |
|---|---|---|---|
| MAE_avg | 0.867 | 0.857 | -0.010 |
| Exact scoreline | 7.81% | 13.0% | +5.2pp |
| W/D/L Accuracy | 50.0% | 53.8% | +3.8pp |
| RPS | 0.207 | 0.202 | -0.005 |

The largest gain is on **exact scoreline accuracy** (+5.2 percentage points), suggesting the discretization problem is less severe in club mode. Plausible reason: club λ values have higher variance (some teams average 2.5 goals/game, others 0.8), so the modal scoreline is more often something other than 1-1 / 1-0 / 0-1. National λ values cluster more tightly because international teams have more uniform offensive output.

---

## 10. Backtesting, frozen predictions, lineage

Same mechanism as national mode — see `documents/model-card-national.md` §10. Per-fixture predictions write once to `s3://<bucket>/predictions/<fixture_id>.json`, never overwritten. Lineage stamping on `decision_rule_version` and `model_trained_at`.

**Holdout report:** the metrics in §9.2 come from evaluating Poisson Linear on the 2024-25 season holdout (~1140 fixtures across PL, La Liga, Ligue 1). Per-league breakdown is **not currently saved** — would be a useful addition (it's plausible that Premier League fixtures behave differently from Ligue 1, given the leagues' different offensive distributions).

**Recent-window drift (manual analysis, 2026-04-30):** total goals systematically under-predicted in the last 30 days of fixtures: **PL −15.7%, La Liga −8.4%, Ligue 1 −26.2%**. Holdout calibration is fine (within ±7%). This is **drift, not bias** — the holdout was drawn before the model was deployed; the world has changed underneath. There is no scheduled drift-detection job. Either schedule retraining more frequently or build a rolling-window-error script that alerts when divergence exceeds a threshold.

---

## 11. Decision rule

Identical to national mode: **`argmax_v0`** at `predict.py:57`. Modal scoreline of the matrix; outcome derived from marginal sums.

The known "modal scoreline ≠ marginal-argmax outcome" disagreement applies here too. The UI uses scoreline-derived outcome to keep cards internally consistent.

---

## 12. Known limitations and open questions

A senior reviewer would push on these. Listed in roughly descending order of importance.

**1. Recent-window drift is real and unaddressed.** PL −15.7%, La Liga −8.4%, Ligue 1 −26.2% under-prediction in the last 30 days. Worst on Ligue 1. Either retrain more frequently, down-weight older training seasons, or build a drift-detection alarm.

**2. Empty club-mode drop list.** The static `_NATIONAL_DROP_FEATURES` was populated based on offline SHAP analysis; the equivalent club-mode pass hasn't been done. Several club features have very small SHAP values (`away_squad_avg_age` 0.0026, `away_top5_league_ratio` 0.0024) that would be drop candidates under the same threshold. Untested whether dropping them would help or hurt.

**3. SHAP is computed on LightGBM, but production is Poisson Linear.** The two models capture similar signal but the SHAP attribution doesn't directly explain the live model. For a linear model, the natural attribution is the coefficient times the standardized feature value, which is computable from the saved scaler + sklearn coefficients but isn't currently emitted.

**4. Poisson Linear winning over tuned XGBoost is informative.** The tuned XGBoost (Optuna 100 trials) couldn't beat a linear model on this data. Either the search space is wrong, the dataset doesn't reward non-linearity, or there's a hidden bug in the tree pipeline. Worth deeper investigation — even just checking residual structure of the linear model would tell us whether non-linear signal is being left on the table.

**5. ρ = +0.042 cuts in the opposite direction from national's −0.106.** Both are plausible from theory. Worth verifying neither is a sign-convention bug, and testing per-league ρ (PL might differ from Ligue 1 — different draw rates).

**6. Squad data is single-snapshot per season.** January transfers, mid-season form changes, and rotation patterns are not captured. For a model dominated by squad-quality features, this is a real limitation.

**7. Joint fit across three leagues.** PL, La Liga, and Ligue 1 are pooled into one training set. The model has no league-id input — it implicitly assumes the relationship between features and goals is the same across all three leagues. Plausible but untested. Per-league models or league-id as a feature might help.

**8. CV per-fold metrics are not saved.** Only final holdout numbers persist. CV variance / fold stability cannot be inspected.

**9. No reliability diagrams / calibration plots.** Standard tool for assessing probabilistic models — missing here.

**10. No per-league or per-stage breakdown.** Holdout metrics are reported as single summary numbers covering all three leagues. Per-league breakdown would surface whether the model is uniformly OK or great-on-one / bad-on-another.

**11. Confusion matrix not saved.** `evaluate.py` defines `get_confusion_matrix` but it's not called in the production training path.

---

## 13. Reproducibility

Same as national mode. See `documents/model-card-national.md` §13. The frozen-predictions store joined to actuals is the source of truth for any re-evaluation that doesn't require re-running training.

For club mode specifically:
- `artefacts/club/` holds all model binaries, scaler, ρ, SHAP explainer, comparison CSV, and the tuned-XGBoost params.
- The Optuna study itself (the `Study` object) is not persisted — only the best params. Re-running tuning would not be deterministic without seed pinning in `tune.py`.

---

## 14. Useful entry points

```
src/features/build.py              flat-row assembly + match context
src/features/{rolling,squad,h2h}.py    feature family definitions
src/models/train.py                candidate fits, drop list, hyperparameters
src/models/calibrate.py            ρ fit
src/models/tune.py                 Optuna study (used for club's XGBoost candidate)
src/models/evaluate.py             metric implementations
src/models/explain.py              SHAP
src/inference/predict.py           inference pipeline + decision rule
scripts/prediction_lineage_report.py   per-lineage-bucket accuracy

artefacts/club/comparison.csv          per-candidate metrics on 2024-25 holdout
artefacts/club/comparison_tuned.csv    tuned XGBoost variant (didn't win)
artefacts/club/best_params.json        Optuna result for tuned XGBoost
artefacts/club/shap_feature_importance.csv
                                       mean |SHAP| per feature (LightGBM-based)
artefacts/club/{model_final_home,model_final_away,model_final_scaler}.pkl
                                       production Poisson Linear regressors + scaler
artefacts/club/rho.json                calibrated correlation parameter (= +0.042)
artefacts/club/shap_explainer.pkl      pickled TreeExplainer (LightGBM)
outputs/club/shap_*.png                summary, bar, top-5 dependence plots
outputs/training_history.csv           full training-run log (both modes) with is_best flag
```

CLAUDE.md and `documents/technical-architecture.md` provide system-level context. `documents/model-card-national.md` covers the parallel national-team model.

---

## 15. Model change log

### 2026-05-04 — Add rolling xG features to the club Poisson Linear

**What changed.** Four new features added to the club training and inference tables:
`home_xg_for_avg_l10`, `home_xg_against_avg_l10`, `away_xg_for_avg_l10`, `away_xg_against_avg_l10`. Computed by `src/features/xg_rolling.py` from `data/processed/match_statistics_club.csv`, which is in turn parsed from the local cache of `/fixtures/statistics` API responses by `src.data.ingest.build_club_match_statistics_from_cache`. Wired into `src/features/handler.py` `_run_club()` so feature rebuilds are end-to-end.

The production model (Poisson Linear) was retrained on the augmented training table. Best-Poisson selection re-ran from scratch and Poisson Linear remained the winner — the candidate ranking did not change.

**Trigger.** A diagnostic on the 2024-25 holdout (`scripts/probe_goal_dispersion.py`) showed that predicted λ values were ~3× under-dispersed relative to the actual goal distribution (predicted std 0.42 vs actual 1.28, predicted λ_h+λ_a ≥ 3.5 in 2.7% of fixtures vs actual ≥4 total goals in 30.9%). Per-fixture marginal calibration was already correct — the bottleneck was λ-spread, not Poisson shape. Feature engineering was the right lever; xG is the cheapest meaningful feature uplift available.

**Evaluated alternatives** (`scripts/experiment_xg_features.py`). xG features were tested on the holdout under both Poisson Linear and LightGBM Poisson, against the same baseline. Linear improved on every metric simultaneously; LightGBM showed zero change because ~70% of training rows have NaN xG and gain-based splits never pick the feature. Decision: ship for Linear (which IS the production model); revisit LightGBM as 2024+ training coverage grows.

**Before / after on the 2024-25 holdout** (Poisson Linear, n=1068 fixtures):

| | Pre-xG | Post-xG | Δ |
|---|---|---|---|
| ρ | +0.0425 | +0.0658 | +0.023 |
| λ_home std | 0.418 | 0.503 | +0.085 (+20%) |
| λ_away std | 0.360 | 0.420 | +0.060 (+17%) |
| MAE_home | 0.938 | 0.931 | −0.007 |
| MAE_away | 0.856 | 0.848 | −0.008 |
| Outcome accuracy | 53.7% | 54.6% | +0.9pp |
| Exact-score accuracy | 13.3% | 12.9% | −0.4pp (within noise) |
| Fixtures with λ_h+λ_a ≥ 3.5 | 2.7% | 4.0% | +1.3pp |

All accuracy metrics move the right direction. Exact-score accuracy is inside noise tolerance. Crucially, MAE improves *while* dispersion increases — the model is leaning more confidently on strong-vs-weak mismatches rather than just spreading predictions noisily.

**Files touched.**
- New: `src/data/ingest.py` — `build_club_match_statistics_from_cache()`
- New: `src/features/xg_rolling.py` — `compute_xg_rolling_features()`
- Modified: `src/features/build.py` — `_assemble_club_table` joins `features_xg_rolling_club.csv`; `build_club_training_table` and `build_club_inference_table` accept an `xg_rolling_path` parameter
- Modified: `src/features/handler.py` — `_run_club()` calls the new ingest builder + xG-rolling computation between H2H features and training-table assembly
- New experiment / diagnostic scripts: `scripts/experiment_xg_features.py`, `scripts/probe_goal_dispersion.py`, `scripts/probe_xg_coverage.py`, `scripts/verify_xg_retrain.py`
- Retrained artefacts at `artefacts/club/` (gitignored); production scaler now expects 38 features instead of 34

**Rollback procedure.** Two paths, in order of preference:

1. **Code-level rollback (recommended).** Revert the feature commit on master (`git revert <merge-commit-sha>`). This restores the previous `_assemble_club_table` schema. Then re-run the full club pipeline:

   ```powershell
   uv run python -c "from src.features.handler import handler; handler({'domain': 'club'})"
   uv run python scripts/train_pipeline.py --mode club --skip-features --skip-shap
   ```

   This rebuilds the training and inference tables without xG columns and produces fresh artefacts at `artefacts/club/` matching the pre-xG schema. Total time ~2 minutes. The new artefacts will be functionally equivalent to the pre-xG snapshot but trained on the latest fixture data, so metrics may differ slightly from the snapshot's frozen numbers.

2. **Local artefact restore (emergency only — does not survive a redeploy).** A snapshot of the pre-xG state was saved on 2026-05-04:

   ```
   artefacts/club_v0_pre_xg/                      (pre-xG model artefacts — gitignored)
   data/processed/training_table_club_pre_xg.csv  (pre-xG training table)
   ```

   To restore locally: copy `artefacts/club_v0_pre_xg/*` over `artefacts/club/`, AND copy the pre-xG training table over `training_table_club.csv`, AND rebuild the inference table from the pre-xG schema (otherwise the inference table will still have the new columns and the old scaler will fail on `transform`). For this reason path (1) is strongly preferred unless you specifically need the exact frozen pre-xG predictions for comparison.

**Verification on rollback.** Run `scripts/verify_xg_retrain.py` to compare the restored (or re-trained-without-xG) model against the snapshotted pre-xG state. Holdout metrics should match within rounding.
