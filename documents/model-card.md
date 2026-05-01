# Model Card — Football Predictions

A senior-data-scientist-oriented assessment of the current production model: what it predicts, how features are built, how the model is trained and selected, how it's evaluated, and what's missing or worth pushing on.

The system fits two independent Poisson regressors on per-fixture engineered features (one for `home_goals`, one for `away_goals`), then combines them via a bivariate-Poisson matrix with a calibrated correlation parameter ρ. Outcome probabilities `P(W/D/L)` are derived by summing regions of the resulting scoreline matrix. The current production pick is **LightGBM Poisson**.

---

## 1. TL;DR — model card summary

| | |
|---|---|
| **Task** | Predict `(home_goals, away_goals)` for international and club football fixtures, with a derived 3×1 outcome distribution |
| **Models trained** | 7 candidates per mode; production picks **LightGBM Poisson** (home + away, two-tower) |
| **Calibration** | Bivariate Poisson with scalar ρ = **−0.1061** (national mode), fit on last 15% of train via Brier-on-draws minimization |
| **Holdout** | WC 2022 (national); most recent completed domestic season (club, currently 2024-25) |
| **Headline metrics (national, holdout)** | MAE_avg **0.867** · RPS **0.207** · W/D/L Accuracy **0.500** · Log loss **0.994** · Brier **0.196** · Exact scoreline acc **7.81%** |
| **Feature count** | 46–48 numeric features post static drop list |
| **Top feature** (mean \|SHAP\|) | `elo_diff` at **0.266** — ~3× the #2 feature |
| **Tuning** | Hard-coded hyperparameters in production. Optuna implemented in `tune.py` but not invoked. |
| **Feature selection** | Static `_NATIONAL_DROP_FEATURES` list (14 features). Variance / correlation / RFECV / permutation pipeline exists in `select.py` but isn't on the prod training path. |
| **Decision rule** | `argmax_v0` — modal scoreline of the bivariate-Poisson matrix; outcome derived from marginal sums |
| **Known weak point** | Exact-scoreline accuracy is poor (~8%, vs. benchmark "Good" 18%). The model collapses onto 1-0 / 0-1 / 1-1 because those are the modes of low-mean Poissons. |

**Honest read of the metrics:** continuous goal expectation is good (MAE beats the 0.85 "Excellent" line; Brier beats "Good"). Discrete outputs — exact scorelines and W/D/L accuracy — are mediocre. Lambda quality is high; the discretization step in the decision rule is the bottleneck.

---

## 2. Target and inputs

**Targets:**
- `home_goals: int ∈ {0, 1, 2, …}` — Poisson regressor
- `away_goals: int ∈ {0, 1, 2, …}` — Poisson regressor (independent fit)
- Derived: `outcome ∈ {away_win, draw, home_win}` from the scoreline matrix

**Inputs:** 46–48 numeric features built per-fixture from the historical-fixture corpus, FIFA rankings, Elo ratings, and squad metadata. No categorical features pass into the model — everything is encoded numerically upstream. No textual or image features.

**Modes:**
- **National**: international fixtures only. Holdout = WC 2022 (`league_id=1, season=2022`). Includes tournament running stats.
- **Club**: Premier League, La Liga, Ligue 1. Holdout = most recent fully completed season. Excludes tournament running stats; includes `rest_days_diff`.

---

## 3. Feature engineering

Six families assembled in `src/features/`, joined into a single flat row per fixture in `build.py`. All features carry a strict **leakage-prevention guard** — derivation uses only data with `date < match_date`, never `≤`.

### 3.1 Rolling form (`rolling.py`)
Per-team windows over prior matches.

| Feature | Definition |
|---|---|
| `goals_scored_avg_l10`, `goals_conceded_avg_l10` | Mean per-match goals over the last 10 prior fixtures |
| `win_rate_l10`, `points_per_game_l10` | Standard form aggregates over last 10 |
| `clean_sheet_rate_l10` | Fraction of last 10 with 0 conceded |
| `form_last5` | Concatenated `W/D/L` string for the last 5 (used for display only; not fed to the model) |
| `matches_available` | Count of prior fixtures actually used (caps at 10) |

**Window = 10 prior fixtures, expanding** until the cap is hit (`rolling.py:21`). Teams with no prior history get `None` and are filled with column median at training time (`train.py:194`). The `matches_available` feature gives the model a way to discount the L10 stats when the window is shallow.

### 3.2 Squad quality (`squad.py`)
Per-`(team_id, season)` aggregates from API-Football's `/players` endpoint.

| Feature | Definition |
|---|---|
| `squad_avg_age` | Mean age across squad |
| `squad_avg_rating` | Mean player rating (0–99). **Set to `None` if <50% of squad has ratings** (`squad.py:60`) — partial-coverage squads are deliberately excluded rather than yielding biased low-N averages |
| `squad_goals_club_season` | Total club-season goals across the squad |
| `top5_league_ratio` | Fraction of squad playing in EPL/La Liga/Bundesliga/Serie A/Ligue 1 |
| `star_player_present` | Boolean — any player rated ≥ 8.0 (`squad.py:26`) |

**Leakage stance:** squad composition for season `S` is treated as known before any fixture in `S`. Mid-season transfers are not modelled.

### 3.3 Head-to-head (`h2h.py`)
Prior-meeting statistics between the two teams.

| Feature | Definition |
|---|---|
| `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_matches_total` | Raw counts from prior H2H |
| `h2h_home_win_rate` | `home_wins / total`, neutral default `0.33` if `< 3` prior meetings (`h2h.py:17,28`) |
| `h2h_home_goals_avg`, `h2h_away_goals_avg` | Per-side scoring rate in prior H2H |
| `h2h_last_winner` | Categorical, most recent prior outcome |

**`min_meetings = 3`** is the threshold below which H2H features fall back to neutral defaults. This avoids creating the spurious signal of a single one-off encounter dominating a team-pair.

Several of these features were dropped post-SHAP as noise — see §4.2.

### 3.4 Tournament running stats (`tournament.py`)
Within-tournament accumulators for international fixtures (national mode only).

| Feature | Definition |
|---|---|
| `matches_played_in_tournament` | Count of completed fixtures by team in current tournament before this match |
| `tournament_goals_scored_so_far`, `tournament_goals_conceded_so_far` | Running goal totals |
| `tournament_yellows_so_far`, `tournament_reds_so_far` | Running card totals |
| `days_since_last_match` | Days since the team's previous fixture in this tournament |
| `came_from_extra_time`, `came_from_shootout` | Booleans for prior match going to AET / penalties |

The accumulator iterates fixtures chronologically per `(league_id, season)` and emits each row's value strictly before the row itself is consumed (`tournament.py:59–100`).

### 3.5 Match context (`build.py:65–77`)
Static metadata known at fixture time.

| Feature | Definition |
|---|---|
| `is_knockout` | Boolean for `round_of_16 / quarterfinal / semifinal / final / third_place` |
| `match_weight` | `stage_weight × competition_weight`. Stage: friendly 0.5, qualifying 0.6, group 0.8, knockout 0.9, final 1.0. Competition: friendly 0.2, qualifying 0.4, World Cup 1.0. |
| `neutral_venue` | Boolean — true for major international tournaments (`league_id ∈ {1, 4, 5, 6, 7, 9, 10}`) |

`match_weight` is dual-purpose: it's both a feature (the model can learn that knockout WC games behave differently from friendlies) **and** a `sample_weight` passed into `fit()` (so a 10-0 friendly doesn't dominate the loss). See §5.

### 3.6 FIFA rankings & Elo (`build.py:85–173`)

| Feature | Definition |
|---|---|
| `home_fifa_rank`, `away_fifa_rank` | Most recent rank with `rank_date <= match_date`. Default 150 for unranked teams. |
| `home_elo`, `away_elo` | Most recent Elo. Default 1300 for new entrants. |

### 3.7 Derived differences (`build.py:279–291`)

| Feature | Definition |
|---|---|
| `form_diff` | `home_points_per_game_l10 − away_points_per_game_l10` |
| `goals_scored_avg_diff` | `home_goals_scored_avg_l10 − away_goals_scored_avg_l10` |
| `rank_diff`, `elo_diff` | Pairwise rank / Elo gaps |
| `squad_rating_diff` | Pairwise squad-rating gap |
| `rest_days_diff` | Club mode only |

These are explicit pairwise differences. A tree model can theoretically learn the same pattern from the parent features, but giving the model the difference directly tends to reduce the depth and number of splits required, which helps with both overfitting and SHAP interpretability.

---

## 4. Feature preprocessing, scaling, and selection

### 4.1 Preprocessing
- **Scaling:** `StandardScaler` is fit on the training set and applied to the **linear** models only (`poisson_linear`, `logistic_regression`). Tree models (XGBoost, LightGBM) use raw features — they're scale-invariant. The scaler is saved to `artefacts/model_final_scaler.pkl` for consistency even when the selected model doesn't use it.
- **Categorical encoding:** none in the model. Outcome labels are encoded with `LabelEncoder` for the classifier branch only (0=away_win, 1=draw, 2=home_win at `train.py:190–191`).
- **Missing-data handling:** features with missing values are filled with the **column median** computed on the training fold (`train.py:194`). There is no separate "is_missing" indicator column. There is no explicit drop threshold — even sparse features are imputed.

### 4.2 Train / test / holdout / calibration boundaries

| Mode | Train | Calibration | Holdout |
|---|---|---|---|
| National | `date < 2022-11-20` | Last 15% of train (chronologically) | WC 2022 (`league_id=1, season=2022`) |
| Club | All seasons before holdout | Last 15% of train | Most recently completed season (currently 2024-25) |

The calibration set is carved out of training data, not from the holdout — so the holdout remains a clean estimate of out-of-tournament performance.

### 4.3 Sample weighting
The `match_weight` column flows into every `fit()` call as `sample_weight=w_train` (`train.py:300, 347–348, 433–434`). Friendly fixtures contribute roughly 5× less than World Cup finals to the training loss. This both reduces noise from low-stakes fixtures and aligns the optimizer with what we actually care about predicting.

### 4.4 Feature selection — what's actually running

**Production:** static drop list at `train.py:59–74`. Fourteen national-mode features were eliminated based on offline SHAP analysis showing mean `|SHAP| < 0.002`:

```
h2h_home_wins, h2h_away_wins, h2h_home_win_rate, h2h_matches_total,
h2h_away_goals_avg, home_top5_league_ratio, away_top5_league_ratio,
top5_ratio_diff, home_tournament_yellows_so_far,
away_tournament_yellows_so_far, home_tournament_reds_so_far,
away_tournament_reds_so_far, away_matches_played_in_tournament,
home_matches_available
```

`_CLUB_DROP_FEATURES` is currently empty (`train.py:77`) — pending a club-specific SHAP pass.

**Dormant infrastructure:** `src/models/select.py` implements a four-stage pipeline (variance threshold → correlation filter → RFECV with `LogisticRegression` + `TimeSeriesSplit(5)` → permutation importance with 30 repeats) that **is not called by the production `train_pipeline.py`**. It's available for offline experiments.

A reviewer should flag this as scope-creep risk: the static list is a snapshot, will go stale, and isn't currently re-derived as part of the training process. A scheduled SHAP re-evaluation (or automating the `select.py` pipeline) would close the loop.

---

## 5. Feature importance

### 5.1 SHAP setup (`src/models/explain.py`)
- **Method:** `shap.TreeExplainer` on the LightGBM home-goals model
- **Background data:** full training set (no separate background sample — TreeExplainer doesn't need one for tree models)
- **Saved:** `artefacts/shap_explainer.pkl`, `outputs/shap_feature_importance.csv`
- **Plots:** summary dot plot, bar importance chart, top-5 dependence plots (`outputs/shap_*.png`)
- **Caveat:** SHAP is computed **only on the home-goals model**. The away-goals model is not explained. For symmetric inputs we'd expect `away_*` features to mirror `home_*`, but this hasn't been verified.

### 5.2 Top features (national mode, mean |SHAP|)

| Rank | Feature | Mean \|SHAP\| | Comment |
|---|---|---|---|
| 1 | `elo_diff` | 0.266 | Dominant signal — ~3× the next feature |
| 2 | `home_squad_goals_club_season` | 0.079 | Club-form prior on national-team players |
| 3 | `match_weight` | 0.057 | Stage × competition (also serves as sample weight) |
| 4 | `rank_diff` | 0.044 | FIFA ranking gap |
| 5 | `home_fifa_rank` | 0.040 | Absolute home ranking |
| 6 | `home_goals_conceded_avg_l10` | 0.037 | Defensive form |
| 7 | `away_fifa_rank` | 0.026 | Absolute away ranking |
| 8 | `away_matches_available` | 0.025 | Form-window depth |
| 9 | `home_squad_avg_age` | 0.024 | Tournament experience proxy |
| 10 | `goals_scored_avg_diff` | 0.021 | Offensive form gap |

**Interpretation notes a reviewer would push on:**

- `elo_diff` carrying ~3× the weight of #2 means most of the model's edge comes from a single (well-engineered) feature. This is expected for football scoreline models — Elo is a summary statistic of historical strength — but it does mean the model is sensitive to how Elo is updated. A stale Elo column would silently degrade everything.
- `rank_diff` (#4) and the absolute-rank features (#5, #7) co-exist. Not collinear by SHAP, but worth checking on a correlation matrix.
- The dependence plots for `elo_diff`, `home_fifa_rank`, `match_weight`, `home_squad_goals_club_season`, `home_squad_avg_age` are saved (`outputs/shap_dependence_*.png`). No automated re-generation; they reflect a single training snapshot.

---

## 6. Model architecture and training

### 6.1 Two-tower bivariate Poisson

```
features ──┬──► LightGBM(objective=poisson) ──► λ_home
           └──► LightGBM(objective=poisson) ──► λ_away

then:  P(h, a) = bivariate_poisson(λ_home, λ_away, ρ)        # 11×11 matrix
       predicted_score   = argmax_{h,a} P(h, a)               # modal scoreline
       p_home_win  = sum_{h>a} P(h, a)
       p_draw       = sum_{h=a} P(h, a)
       p_away_win   = sum_{h<a} P(h, a)
```

The two towers are fit **independently** with **identical hyperparameters**. Joint estimation (single model with two outputs) was not tried; it would force shared regularization across what are arguably two different prediction tasks (a team's offence and its opponent's defence interact differently from a team's defence and its opponent's offence).

### 6.2 Hyperparameters (production, `train.py:415–446`)

```python
LGBMRegressor(
    objective="poisson",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
)
.fit(X_tr, y_tr,
     sample_weight=w_tr,
     eval_set=[(X_val, y_val)],     # last 20% of train
     callbacks=[early_stopping(50)],
)
```

Modest depth, modest tree count, aggressive early stopping — defensive choices for a small dataset (~26k national rows). Subsample / colsample at 0.8 add stochastic regularization.

### 6.3 Why LightGBM Poisson over the others

The full candidate set is evaluated on the holdout, sorted by `MAE_avg`:

| Model | MAE_avg | Train time |
|---|---|---|
| **LightGBM Poisson** ← selected | **0.867** | 0.26 s |
| XGBoost Poisson | 0.906 | (slower) |
| Poisson Linear (sklearn `PoissonRegressor`) | 0.930 | (fast) |
| FIFA-rank-only Poisson (baseline) | (worse) | — |
| Mean-goals (floor baseline) | (worst) | — |

The XGBoost Classifier (`multi:softprob` directly on W/D/L) gets 48.4% accuracy vs. the Poisson-derived 50%, confirming the choice to predict goals and derive outcomes rather than predict outcomes directly.

### 6.4 Calibration — bivariate Poisson ρ (`src/models/calibrate.py`)

The two independent Poissons systematically misprice draws. The fix is a single scalar `ρ` that re-weights the diagonal of the scoreline matrix.

- **Optimizer:** `scipy.optimize.minimize_scalar` with `method="bounded"`, bounds `(-0.5, 0.5)` (`calibrate.py:91`)
- **Loss:** Brier score on draw probability — `mean((p_draw − is_draw)²)` (`calibrate.py:84–89`)
- **Fit set:** last 15% of train (calibration carve-out, separate from holdout)
- **Saved:** `artefacts/rho.json` — `{"rho": -0.10609...}`

**The current ρ = −0.106 is mildly negative.** This implies the calibration set has *fewer* draws than independence predicts. That cuts against the literature's usual finding (positive correlation, more draws than independence — the canonical Dixon-Coles motivation).

A reviewer should:
1. **Verify the sign convention** — `_bivariate_poisson_matrix` in `predict.py` should be checked to confirm a negative ρ deflates the diagonal as intended.
2. **Check stability** — does ρ flip sign across CV folds, modes, or retrains? Currently it's a single fitted scalar with no confidence interval.
3. **Decide whether per-mode (national vs club) or per-stage (group vs knockout) calibration would help** — currently it's a single ρ shared across all fixtures in a mode.

---

## 7. Hyperparameter tuning

`src/models/tune.py` implements an Optuna study with:
- **Search space:** `n_estimators ∈ [100, 1000]`, `learning_rate ∈ [0.01, 0.3]` (log), `max_depth ∈ [3, 8]`, `min_child_weight ∈ [1, 10]`, `subsample` and `colsample_bytree ∈ [0.6, 1.0]`, `gamma ∈ [0, 5]`, `reg_alpha` / `reg_lambda ∈ [1e-8, 10]` (log)
- **Objective:** mean MAE across `TimeSeriesSplit(5)` folds for home + away
- **Budget:** `n_trials=100`, `timeout=3600s`
- **Output:** `best_params.json`

**It is not called by the production training pipeline.** Production hyperparameters are hard-coded at `train.py:415–421`. The Optuna code is offline scaffolding.

For a reviewer, this is the second-most important "implemented but not used" gap (after `select.py`). The fixed-hyperparameters choice is reasonable for a small dataset where over-aggressive tuning overfits the holdout, but it should be either run once and the resulting parameters frozen explicitly, or re-run on a schedule.

---

## 8. Cross-validation

| Setting | Value |
|---|---|
| Splitter | `sklearn.model_selection.TimeSeriesSplit` |
| `n_splits` | 5 |
| `gap` | 0 |
| `max_train_size` | None (expanding window) |
| Scoring | model-dependent (Poisson-deviance via early-stopping `eval_set`; neg log loss in RFECV; MAE in Optuna) |

Used for: early-stopping the boosters on the inner validation fold (last 20% of train), and for the dormant `select.py` and `tune.py` pipelines.

**Per-fold metrics are not saved.** Only the final holdout numbers land in `outputs/model_comparison.csv`. A reviewer wanting CV variance / fold stability would need to instrument this.

---

## 9. Evaluation metrics

### 9.1 Computed metrics (`src/models/evaluate.py`)

| Metric | Definition | Code |
|---|---|---|
| `MAE_home`, `MAE_away`, `MAE_avg` | `MAE(y_true, round(λ))` per side, then averaged | `evaluate.py:86–88` |
| Exact scoreline accuracy | `mean(argmax(P) == (y_home, y_away))` | `evaluate.py:40–55` |
| Ranked Probability Score (RPS) | Cumulative-distribution squared difference across `{away, draw, home}` | `evaluate.py:58–74` |
| Outcome accuracy | `argmax(p_a, p_d, p_h) == truth` | `evaluate.py:94` |
| Log loss | Multinomial cross-entropy with clipped probabilities | `evaluate.py:96–99` |
| Brier score | Mean squared error per class, averaged | `evaluate.py:101–106` |

### 9.2 Achieved values vs. benchmark targets

National mode, LightGBM Poisson, evaluated on WC 2022 holdout:

| Metric | Naive | Good | Excellent | **Achieved** | Verdict |
|---|---|---|---|---|---|
| MAE (goals) | 1.20 | 0.95 | 0.85 | **0.867** | Beats Excellent |
| Exact scoreline acc | 10% | 18% | 25% | **7.81%** | **Worse than Naive** |
| RPS | 0.24 | 0.20 | 0.17 | **0.207** | Just under Good |
| W/D/L Accuracy | 45% | 52% | 57% | **50.0%** | Below Good |
| Log loss | 1.05 | 0.95 | 0.88 | **0.994** | Just over Good |
| Brier | 0.24 | 0.21 | 0.19 | **0.196** | Beats Good |

### 9.3 What the numbers say

The headline tension is between **continuous** (MAE, Brier — strong) and **discrete** (exact scoreline accuracy, W/D/L accuracy — weak).

This is consistent with a model that produces well-calibrated `λ` values but loses information at discretization. With λ_home ≈ 1.3 and λ_away ≈ 1.1 — typical values — the modal scoreline of an independent Poisson is **always** 1-0, 1-1, or 0-1, regardless of the actual lambdas. So the model can have great `MAE` (its λs are right) and terrible exact-scoreline accuracy (its argmax always falls in the same handful of low-score buckets).

A previous experiment (PR #10) attempted to replace `argmax` with a rounded-expected-goals + outcome-consistency rule, which would shift typical predictions to 2-1 / 1-2 / 3-1. The user evaluated the simulation and rejected the change — the new dominant pattern wasn't convincing either. **The current rule is `argmax_v0` and the project is explicitly not re-proposing rounded-expected without new evidence.** This is documented in memory.

---

## 10. Backtesting, frozen predictions, and lineage

### 10.1 Frozen-prediction pattern

Every prediction is **written exactly once** to `s3://<bucket>/predictions/<fixture_id>.json`. Re-runs of the inference Lambda **do not overwrite** existing files. This is the most consequential operational property of the prediction layer because it makes accuracy reporting honest — when the recent-results card says "we predicted 2-1", that's literally the prediction that was made *before* kickoff, not a back-fitted view from a later model.

### 10.2 Lineage stamping

Each frozen prediction carries:
- `decision_rule_version` — string, currently `"argmax_v0"`. Bumped when the decision rule changes.
- `model_trained_at` — ISO timestamp from the artefact `mtime`. Bumped on every retrain.
- `prediction_made_at`, `backfill` (boolean), λs, scoreline, outcome probs.

### 10.3 Lineage report

`scripts/prediction_lineage_report.py` groups frozen predictions by `(decision_rule_version, model_trained_at)`, joins to actuals, and reports per-bucket: `n`, outcome accuracy, scoreline accuracy, MAE_home, MAE_away.

This makes "did the new model improve things?" answerable apples-to-apples without contaminating the comparison with predictions written under the old model.

**Current report state:** mostly one bucket — `argmax_v0` / unknown — because lineage stamping only landed in PR #11 (2026-04-30) and frozen predictions written before that don't carry the metadata. Future buckets will populate as retrains happen.

---

## 11. Known limitations and open questions

A senior reviewer would push on these. They're listed in roughly descending order of importance.

**1. Discretization is the bottleneck.**
The lambda quality is good (MAE_avg 0.867 beats Excellent). The decision rule throws information away. Worth investigating: probability-weighted scoreline (weighted average rounded), expected-goals + draw-adjustment, or just publishing top-3 most likely scorelines instead of one.

**2. ρ is mildly negative — verify the sign and consider stratification.**
The literature's prior is that football has positive within-fixture goal correlation (more draws than independence predicts). A negative ρ on the calibration set is unusual — could be real, could be a sign convention bug. Worth: (a) plotting actual vs. predicted draw rate by holdout league/stage, (b) checking ρ stability across CV folds, (c) trying per-mode and per-stage ρ.

**3. Recent-window drift is not automated.**
Manual analysis (as of 2026-04-30) showed total goals systematically under-predicted in recent fixtures: PL −15.7%, La Liga −8.4%, Ligue 1 −26.2%. Holdout calibration is fine (within ±7%). This is **drift, not bias** — the holdout was drawn before the model was deployed, so the world has changed underneath. There is no scheduled drift-detection job; the analysis was ad-hoc. Either schedule retraining more frequently or build a rolling-window-error script that alerts when divergence exceeds a threshold.

**4. Optuna and `select.py` are dormant.**
Both implemented, neither in the production training path. Either commit to running them on a schedule, or delete the unused infrastructure to reduce cognitive load. Both — running once and freezing the result — is a reasonable middle ground if the dataset isn't growing fast.

**5. SHAP is computed on the home-goals model only.**
The away-goals model has no explainability artefacts. For a symmetric feature set, the away SHAP should mostly mirror the home SHAP, but this is unverified. Also, the SHAP report is a single snapshot; there's no scheduled re-run.

**6. CV per-fold metrics are not saved.**
Only the final holdout numbers persist (`outputs/model_comparison.csv`). A reviewer asking for fold variance, training-time stability, or CV-vs-holdout gap analysis cannot answer those questions from current artefacts.

**7. No reliability diagrams / calibration plots.**
The code that would generate them isn't there. For a probabilistic model, this is the standard assess-the-calibration tool. Recommend adding a per-mode reliability plot (binned predicted probability vs. binned empirical frequency for each outcome class).

**8. No per-league or per-stage breakdown.**
Holdout metrics are reported as single summary numbers. Knowing the model is good at WC group stage but bad at WC knockouts would be actionable. The data exists; it's just not aggregated this way.

**9. No confusion matrix in the standard evaluation flow.**
`evaluate.py:197–204, 207–221` defines `get_confusion_matrix` and `get_classification_report` but they aren't called in `train_pipeline.py`. Surfacing the confusion matrix would directly diagnose the W/D/L accuracy weak point.

**10. Squad data is single-snapshot per season.**
Mid-season transfers, injuries-of-the-day, and rotation patterns are not modelled. A senior reviewer would push on whether this is actually a blocker for international tournaments (where squads are picked once per tournament) — probably not — but it's a real limitation for club-mode predictions.

**11. Joint vs. independent two-tower fit.**
Whether shared representation across home and away towers would help is open. Likely not — different conditional distributions — but untested.

---

## 12. Reproducibility

| | |
|---|---|
| Random seeds | LightGBM uses default seed; not pinned in code. CV folds are deterministic via `TimeSeriesSplit`. |
| Data versioning | Raw API JSONs are immutable in S3 (date-prefixed). Training/inference Parquets are regenerated by the feature Lambda; no DVC / explicit versioning of derived tables. |
| Artefact versioning | `artefacts/` is overwritten on each retrain. The frozen-prediction lineage (`model_trained_at`) is the canonical "which model" pointer for a given prediction. No archived past models. |
| Environment | Lambda Docker images pin via image tag; `pyproject.toml` pins via `uv lock`. Dependencies are reproducible; the trained model itself is not byte-identical across retrains because LightGBM seed is unset. |

For a reviewer who wants a reproducible re-evaluation: the **frozen predictions** are the source of truth. Even without re-running the training pipeline, `predictions/<fid>.json` joined to actuals gives an honest accuracy view per lineage bucket.

---

## 13. Useful entry points for a reviewer

```
src/features/build.py              flat-row assembly + match context
src/features/{rolling,squad,h2h,tournament}.py
                                   feature family definitions
src/models/train.py                candidate fits, drop list, hyperparameters
src/models/calibrate.py            ρ fit
src/models/select.py               (dormant) feature-selection pipeline
src/models/tune.py                 (dormant) Optuna study
src/models/evaluate.py             metric implementations
src/models/explain.py              SHAP
src/inference/predict.py           inference pipeline + decision rule
scripts/prediction_lineage_report.py   per-lineage-bucket accuracy

outputs/model_comparison.csv       per-mode candidate metrics on holdout
outputs/shap_feature_importance.csv
                                   mean |SHAP| per feature, sorted
outputs/shap_*.png                 summary, bar, top-5 dependence plots
artefacts/{model_final_home,model_final_away,model_final_scaler}.pkl
artefacts/rho.json                 calibrated correlation parameter
artefacts/shap_explainer.pkl       pickled TreeExplainer
```

CLAUDE.md and `documents/technical-architecture.md` provide system-level context.
