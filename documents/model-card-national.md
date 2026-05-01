# Model Card — National Teams

A senior-data-scientist-oriented assessment of the **national-team** mode of the football-predictions system. National mode targets international fixtures (World Cup, EURO, Nations League, AFCON, Copa America, Gold Cup, Asian Cup, friendlies). The headline target is **WC 2026** (June–July 2026).

The system fits two independent Poisson regressors on per-fixture engineered features (one for `home_goals`, one for `away_goals`), then combines them via a bivariate-Poisson matrix with a calibrated correlation parameter ρ. Outcome probabilities `P(W/D/L)` are derived by summing regions of the resulting scoreline matrix.

For the **club** counterpart of this card see `documents/model-card-clubs.md`. For system architecture see `documents/technical-architecture.md`.

---

## 1. TL;DR — model card summary

| | |
|---|---|
| **Task** | Predict `(home_goals, away_goals)` for international fixtures, with a derived 3×1 outcome distribution |
| **Holdout** | WC 2022 (`league_id=1, season=2022`) — fixtures `date >= 2022-11-20` |
| **Production pick** | **LightGBM Poisson**, two-tower (home + away regressors fit independently) |
| **Calibration** | Bivariate Poisson with scalar ρ = **−0.1061**, fit on last 15% of train via Brier-on-draws minimization |
| **Headline metrics (holdout)** | MAE_avg **0.867** · RPS **0.207** · W/D/L Accuracy **0.500** · Log loss **0.994** · Brier **0.196** · Exact scoreline acc **7.81%** |
| **Feature count** | 46 numeric features (post static drop of 14) |
| **Top feature** (mean \|SHAP\|) | `elo_diff` at **0.266** — ~3.4× the #2 feature |
| **Tuning** | Hard-coded hyperparameters in production. Optuna implemented in `tune.py` but not invoked for national mode. |
| **Feature selection** | Static `_NATIONAL_DROP_FEATURES` list (14 features) at `train.py:59–74` |
| **Decision rule** | `argmax_v0` — modal scoreline of the bivariate-Poisson matrix; outcome derived from marginal sums |
| **Known weak point** | Exact-scoreline accuracy is poor (7.81%, vs benchmark "Good" 18%). The model collapses onto 1-0 / 0-1 / 1-1 because those are the modes of low-mean Poissons. |

**Honest read of the metrics:** continuous goal expectation is good (MAE beats the 0.85 "Excellent" line; Brier beats "Good"). Discrete outputs — exact scorelines and W/D/L accuracy — are mediocre. Lambda quality is high; the discretization step in the decision rule is the bottleneck.

---

## 2. Target and inputs

**Targets:**
- `home_goals: int ∈ {0, 1, 2, …}` — Poisson regressor
- `away_goals: int ∈ {0, 1, 2, …}` — Poisson regressor (independent fit)
- Derived: `outcome ∈ {away_win, draw, home_win}` from the scoreline matrix

**Inputs:** 46 numeric features built per-fixture from the historical international-fixture corpus, FIFA rankings, Elo ratings, and squad metadata. No categorical features pass into the model — everything is encoded numerically upstream. No textual or image features.

**Scope:** all international fixtures of 1990 onward — World Cup, EURO, Nations League, AFCON, Copa America, Gold Cup, Asian Cup, plus friendlies from 2010. **Including friendlies** is deliberate: the fixture corpus would be too small without them, but they're down-weighted via `match_weight` to prevent low-stakes matches dominating the loss (see §4.3).

---

## 3. Feature engineering

Five families assembled in `src/features/`, joined into a single flat row per fixture in `build.py`. All features carry a strict **leakage-prevention guard** — derivation uses only data with `date < match_date`, never `≤`.

### 3.1 Rolling form (`rolling.py`)
Per-team windows over prior international matches.

| Feature | Definition |
|---|---|
| `goals_scored_avg_l10`, `goals_conceded_avg_l10` | Mean per-match goals over the last 10 prior fixtures |
| `win_rate_l10`, `points_per_game_l10` | Standard form aggregates over last 10 |
| `clean_sheet_rate_l10` | Fraction of last 10 with 0 conceded |
| `form_last5` | Concatenated `W/D/L` string for the last 5 (display only; not fed to the model) |
| `matches_available` | Count of prior fixtures actually used (caps at 10) |

Window = 10 prior fixtures, expanding until the cap is hit (`rolling.py:21`). Teams with no prior history get `None` and are filled with column median at training time (`train.py:194`). The `matches_available` feature gives the model a way to discount the L10 stats when the window is shallow — relevant for newly-emerging international teams.

### 3.2 Squad quality (`squad.py`)
Per-`(team_id, season)` aggregates from API-Football's `/players` endpoint. For national teams the "season" maps to a tournament cycle (e.g. WC 2022 squad).

| Feature | Definition |
|---|---|
| `squad_avg_age` | Mean age across squad |
| `squad_avg_rating` | Mean player rating (0–99). Set to `None` if <50% of squad has ratings (`squad.py:60`) |
| `squad_goals_club_season` | Total club-season goals across the squad — captures players' form at their club level going into national duty |
| `top5_league_ratio` | Fraction of squad playing in EPL/La Liga/Bundesliga/Serie A/Ligue 1 |
| `star_player_present` | Boolean — any player rated ≥ 8.0 (`squad.py:26`) |

For national-mode squads, `top5_league_ratio` is mostly informative for non-European teams (where it's a proxy for "how many players have top-5 league experience"). For European teams it tends to saturate near 1.0.

**Leakage stance:** squad composition for tournament `T` is treated as known before any fixture in `T`. This is realistic — squads are announced before tournaments start.

### 3.3 Head-to-head (`h2h.py`)
Prior-meeting statistics between the two national teams.

| Feature | Definition |
|---|---|
| `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_matches_total` | Raw counts from prior H2H |
| `h2h_home_win_rate` | Default `0.33` if `< 3` prior meetings (`h2h.py:17,28`) |
| `h2h_home_goals_avg`, `h2h_away_goals_avg` | Per-side scoring rate in prior H2H |
| `h2h_last_winner` | Categorical, most recent prior outcome |

`min_meetings = 3` is the threshold below which H2H features fall back to neutral defaults. International H2H records are often sparse (e.g. Wales vs Iran has few prior meetings) — without the threshold, single one-off encounters would create spurious signal.

**Most H2H features get dropped** post-SHAP — see §4.2.

### 3.4 Tournament running stats (`tournament.py`) — **national-only**
Within-tournament accumulators. This family is unique to national mode.

| Feature | Definition |
|---|---|
| `matches_played_in_tournament` | Count of completed fixtures by team in current tournament before this match |
| `tournament_goals_scored_so_far`, `tournament_goals_conceded_so_far` | Running goal totals |
| `tournament_yellows_so_far`, `tournament_reds_so_far` | Running card totals |
| `days_since_last_match` | Days since the team's previous fixture in this tournament |
| `came_from_extra_time`, `came_from_shootout` | Booleans for prior match going to AET / penalties |

The accumulator iterates fixtures chronologically per `(league_id, season)` and emits each row's value strictly before the row itself is consumed (`tournament.py:59–100`). Rationale: international tournaments have strong sequencing effects — a team that just played 120 minutes plus penalties is materially different from a team coming off 90 minutes.

The cards (`tournament_yellows_so_far`, `tournament_reds_so_far`) are dropped post-SHAP — they don't carry signal at the model's resolution.

### 3.5 Match context (`build.py:65–77`)

| Feature | Definition |
|---|---|
| `is_knockout` | Boolean for `round_of_16 / quarterfinal / semifinal / final / third_place` |
| `match_weight` | `stage_weight × competition_weight`. Stage: friendly 0.5, qualifying 0.6, group 0.8, knockout 0.9, final 1.0. Competition: friendly 0.2, qualifying 0.4, World Cup 1.0. |
| `neutral_venue` | Boolean — true for major international tournaments (`league_id ∈ {1, 4, 5, 6, 7, 9, 10}`) |

`match_weight` is dual-purpose: it's both a feature (the model can learn that knockout WC games behave differently from friendlies) **and** a `sample_weight` passed into `fit()` (so a 10-0 friendly doesn't dominate the loss). In national mode it's particularly load-bearing because the corpus mixes WC finals with January friendlies.

### 3.6 FIFA rankings & Elo (`build.py:85–173`)

| Feature | Definition |
|---|---|
| `home_fifa_rank`, `away_fifa_rank` | Most recent rank with `rank_date <= match_date`. Default 150 for unranked teams. |
| `home_elo`, `away_elo` | Most recent Elo. Default 1300 for new entrants. |

For national teams, FIFA rank and Elo are the two most informative absolute-strength features. They're well-correlated but capture slightly different information — Elo reflects recent results more aggressively; FIFA rank uses a longer averaging window.

### 3.7 Derived differences (`build.py:279–291`)

| Feature | Definition |
|---|---|
| `form_diff` | `home_points_per_game_l10 − away_points_per_game_l10` |
| `goals_scored_avg_diff` | `home_goals_scored_avg_l10 − away_goals_scored_avg_l10` |
| `rank_diff`, `elo_diff` | Pairwise rank / Elo gaps |
| `squad_rating_diff` | Pairwise squad-rating gap |
| `top5_ratio_diff` | Pairwise top-5-league-ratio gap |

These are explicit pairwise differences. A tree model can theoretically learn the same pattern from the parent features, but giving the model the difference directly tends to reduce the depth and number of splits required, which helps with both overfitting and SHAP interpretability. **`elo_diff` ends up the dominant feature** (see §5.2).

---

## 4. Feature preprocessing, splits, selection

### 4.1 Preprocessing
- **Scaling:** `StandardScaler` is fit on the training set and applied to the **linear** models only (`poisson_linear`, `logistic_regression`). Tree models (XGBoost, LightGBM) use raw features. The scaler is saved to `artefacts/model_final_scaler.pkl` for consistency even when the selected model doesn't use it.
- **Categorical encoding:** none in the model. Outcome labels are encoded with `LabelEncoder` for the classifier branch only (0=away_win, 1=draw, 2=home_win at `train.py:190–191`).
- **Missing-data handling:** features with missing values are filled with the **column median** computed on the training fold (`train.py:194`). No "is_missing" indicator column. No drop threshold — even sparse features are imputed.

### 4.2 Train / test / calibration boundaries

| | National mode |
|---|---|
| Train | All fixtures with `date < 2022-11-20` |
| Calibration carve-out | Last 15% of train, chronologically |
| Holdout (test) | WC 2022 (`league_id=1, season=2022`), all knockout + group fixtures |

The calibration set is carved out of training data, not from the holdout — so the holdout remains a clean estimate of out-of-tournament performance.

### 4.3 Sample weighting
The `match_weight` column flows into every `fit()` call as `sample_weight=w_train` (`train.py:300, 347–348, 433–434`). Friendly fixtures contribute roughly 5× less than World Cup finals to the training loss. This both reduces noise from low-stakes fixtures and aligns the optimizer with what we actually care about predicting. **For national mode this is particularly important** because the corpus is half friendlies by row count.

### 4.4 Feature selection

**Production:** static drop list at `train.py:59–74`. Fourteen features were eliminated based on offline SHAP analysis showing mean `|SHAP| < 0.002`:

```
h2h_home_wins, h2h_away_wins, h2h_home_win_rate, h2h_matches_total,
h2h_away_goals_avg, home_top5_league_ratio, away_top5_league_ratio,
top5_ratio_diff, home_tournament_yellows_so_far,
away_tournament_yellows_so_far, home_tournament_reds_so_far,
away_tournament_reds_so_far, away_matches_played_in_tournament,
home_matches_available
```

**Dormant infrastructure:** `src/models/select.py` implements a four-stage pipeline (variance threshold → correlation filter → RFECV with `LogisticRegression` + `TimeSeriesSplit(5)` → permutation importance with 30 repeats) that **is not called by the production training path**. It's available for offline experiments.

A reviewer should flag this as scope-creep risk: the static list is a snapshot, will go stale, and isn't currently re-derived as part of the training process. A scheduled SHAP re-evaluation (or automating the `select.py` pipeline) would close the loop.

---

## 5. Feature importance

### 5.1 SHAP setup (`src/models/explain.py`)
- **Method:** `shap.TreeExplainer` on the LightGBM home-goals model
- **Background data:** full training set
- **Saved:** `artefacts/shap_explainer.pkl`, `artefacts/shap_feature_importance.csv`
- **Plots:** `outputs/shap_summary_home_win.png`, `outputs/shap_bar_importance.png`, and dependence plots for the top 5 features
- **Caveat:** SHAP is computed only on the home-goals model; the away-goals model has no explainability artefacts.

### 5.2 Top features (mean |SHAP|)

| Rank | Feature | Mean \|SHAP\| | Comment |
|---|---|---|---|
| 1 | `elo_diff` | **0.266** | Dominant signal — ~3.4× the next feature |
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

- `elo_diff` carrying ~3.4× the weight of #2 means most of the model's edge comes from a single (well-engineered) feature. Expected for football scoreline models — Elo is a summary statistic of historical strength — but it does mean the model is sensitive to how Elo is updated. A stale Elo column would silently degrade everything.
- `rank_diff` (#4) and the absolute-rank features (#5, #7) co-exist. Not collinear by SHAP, but worth checking on a correlation matrix.
- The dependence plots for `elo_diff`, `home_fifa_rank`, `match_weight`, `home_squad_goals_club_season`, `home_squad_avg_age` are saved (`outputs/shap_dependence_*.png`). No automated re-generation; they reflect a single training snapshot.
- **Tournament running stats (`matches_played_in_tournament`, etc.) do not appear in the top 10**, despite being a national-specific feature family. Either the signal is real but small (a few percentage points of accuracy on knockouts), or the training data has too few in-tournament matches to learn the pattern.

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

The two towers are fit **independently** with **identical hyperparameters**.

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

Modest depth, modest tree count, aggressive early stopping — defensive choices for a small dataset (~26k national rows). Subsample / colsample at 0.8 add stochastic regularization. **These hyperparameters are NOT Optuna-tuned for national mode** — they're the train.py defaults.

### 6.3 Why LightGBM Poisson (full candidate comparison, holdout WC 2022)

Sorted by `MAE_avg`, lower is better:

| Model | MAE_home | MAE_away | MAE_avg | Acc | RPS | Log loss | Brier | Train-time |
|---|---|---|---|---|---|---|---|---|
| **LightGBM Poisson** ← selected | 0.953 | 0.781 | **0.867** | 0.500 | 0.207 | 0.994 | 0.196 | 0.36s |
| XGBoost Poisson | 1.000 | 0.812 | 0.906 | 0.500 | 0.208 | 0.992 | 0.196 | 0.83s |
| Poisson Linear (sklearn) | 1.000 | 0.859 | 0.930 | 0.531 | 0.215 | 1.017 | 0.202 | 0.01s |
| Baseline rank-only Poisson | 1.047 | 0.828 | 0.938 | 0.484 | 0.215 | 1.018 | 0.203 | 2.1s |
| Baseline mean-goals | 1.109 | 0.828 | 0.969 | 0.453 | 0.232 | 1.065 | 0.215 | 0s |
| XGBoost Classifier | — | — | — | 0.484 | 0.214 | 1.037 | 0.205 | 0.86s |
| Logistic Regression | — | — | — | 0.391 | 0.235 | 1.140 | 0.226 | 0.05s |

LightGBM wins on `MAE_avg` (the model selection criterion). Notably, **Poisson Linear has higher accuracy (53.1%)** but worse MAE — the linear's W/D/L decisions happen to be slightly better while its goal expectations are slightly worse. The selection rule prioritizes MAE.

The XGBoost Classifier (`multi:softprob` directly on W/D/L) gets 48.4% accuracy vs the Poisson-derived 50%, supporting the choice to predict goals and derive outcomes rather than predict outcomes directly.

### 6.4 Calibration — bivariate Poisson ρ (`src/models/calibrate.py`)

The two independent Poissons systematically misprice draws. The fix is a single scalar `ρ` that re-weights the diagonal of the scoreline matrix.

- **Optimizer:** `scipy.optimize.minimize_scalar` with `method="bounded"`, bounds `(-0.5, 0.5)` (`calibrate.py:91`)
- **Loss:** Brier score on draw probability — `mean((p_draw − is_draw)²)` (`calibrate.py:84–89`)
- **Fit set:** last 15% of train (calibration carve-out, separate from holdout)
- **Saved:** `artefacts/rho.json` — `{"rho": -0.10609858...}`

**The current ρ = −0.106 is mildly negative.** This implies the calibration set has *fewer* draws than independence predicts. That cuts against the literature's usual finding for football (positive correlation, more draws than independence — the canonical Dixon-Coles motivation).

A reviewer should:
1. **Verify the sign convention** — `_bivariate_poisson_matrix` in `predict.py` should be checked to confirm a negative ρ deflates the diagonal as intended.
2. **Check stability** — does ρ flip sign across CV folds, or across retrains? Currently it's a single fitted scalar with no confidence interval.
3. **Note that the club-mode card has ρ = +0.042** — opposite sign. Either the two corpora genuinely have different draw correlation structures (plausible — international tournaments have more "must-not-lose" knockout draws than league play, but also more thrashings of weak teams), or one of the two is fitting noise.

---

## 7. Hyperparameter tuning

`src/models/tune.py` implements an Optuna study for XGBoost Poisson. **It is not invoked for national mode in the production pipeline.** No `artefacts/best_params.json` exists for national — only for club. The national hyperparameters are the train.py defaults at lines 415–421.

The fixed-hyperparameters choice is reasonable for a small dataset where over-aggressive tuning overfits the holdout, but a reviewer should note: there's no documented rationale for the specific values (500 trees, lr 0.05, depth 5). They're defensible defaults but not tuned to this dataset.

---

## 8. Cross-validation

| Setting | Value |
|---|---|
| Splitter | `sklearn.model_selection.TimeSeriesSplit` |
| `n_splits` | 5 |
| `gap` | 0 |
| `max_train_size` | None (expanding window) |

Used for: early-stopping the boosters on the inner validation fold (last 20% of train). Per-fold metrics are not saved — only the final holdout numbers land in `artefacts/comparison.csv` and `outputs/training_history.csv`. A reviewer wanting CV variance / fold stability would need to instrument this.

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

### 9.2 Achieved values vs benchmark targets

LightGBM Poisson, evaluated on WC 2022 holdout:

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

This is consistent with a model that produces well-calibrated `λ` values but loses information at discretization. With λ_home ≈ 1.3 and λ_away ≈ 1.1 — typical national-mode values — the modal scoreline of an independent Poisson is **always** 1-0, 1-1, or 0-1, regardless of the actual lambdas. So the model can have great `MAE` (its λs are right) and terrible exact-scoreline accuracy (its argmax always falls in the same handful of low-score buckets).

A previous experiment (PR #10, closed) attempted to replace `argmax` with a rounded-expected-goals + outcome-consistency rule, which would shift typical predictions to 2-1 / 1-2 / 3-1. The user evaluated the simulation and rejected the change — the new dominant pattern wasn't convincing either. **The current rule is `argmax_v0` and the project is explicitly not re-proposing rounded-expected without new evidence.**

---

## 10. Backtesting, frozen predictions, lineage

### 10.1 Frozen-prediction pattern

Every prediction is **written exactly once** to `s3://<bucket>/predictions/<fixture_id>.json`. Re-runs of the inference Lambda **do not overwrite** existing files. This makes accuracy reporting honest — when the recent-results card says "we predicted 2-1", that's literally the prediction that was made *before* kickoff.

### 10.2 Lineage stamping

Each frozen prediction carries `decision_rule_version` (currently `"argmax_v0"`), `model_trained_at` (ISO timestamp from artefact mtime), `prediction_made_at`, `backfill` (boolean), λs, scoreline, outcome probs.

`scripts/prediction_lineage_report.py` groups frozen predictions by `(decision_rule_version, model_trained_at)` and reports per-bucket accuracy. The current report (`outputs/prediction_lineage_report.csv`) has mostly one bucket — `argmax_v0` / unknown — because lineage stamping landed in PR #11 (2026-04-30) and frozen predictions written before that don't carry the metadata.

### 10.3 WC 2022 holdout report

The selected LightGBM Poisson model produces the metrics in §9.2 on the WC 2022 holdout (64 fixtures: 48 group + 16 knockout). Per-stage breakdown (group vs knockout) is **not currently saved** — would be a useful addition.

---

## 11. Decision rule

Current algorithm: **`argmax_v0`** at `predict.py:57`.

1. Predict λ_home, λ_away from feature vector (`predict.py:154–155`, clipped `[0.01, 10.0]`)
2. Compute bivariate Poisson scoreline matrix using ρ (`predict.py:162`)
3. `predicted_score = argmax_{h,a}` of the matrix (`predict.py:163–164`)
4. `p_home_win = sum_{h>a}`, `p_draw = sum_{h=a}`, `p_away_win = sum_{h<a}` (`predict.py:165–167`)
5. `predicted_outcome = argmax(p_h, p_d, p_a)` (`predict.py:177`)

**Known nuance:** the modal scoreline (step 3) and the marginal-argmax outcome (step 5) can disagree. The UI deliberately derives the displayed outcome from the predicted scoreline rather than from the W/D/L marginals to keep the cards internally consistent.

---

## 12. Known limitations and open questions

A senior reviewer would push on these. Listed in roughly descending order of importance.

**1. Discretization is the bottleneck.** Lambda quality is good (MAE_avg 0.867 beats Excellent). The decision rule throws information away. Worth investigating: probability-weighted scoreline, expected-goals + draw-adjustment, or just publishing top-3 most likely scorelines instead of one.

**2. Negative ρ deserves verification.** Football literature's prior is that within-fixture goal correlation is positive (more draws than independence predicts). A negative ρ on the calibration set is unusual. Could be real (international fixtures have a high-variance distribution mixing thrashings with cagey knockouts), could be a sign-convention bug. The club-mode ρ is +0.042 — opposite sign. Worth investigating both.

**3. WC 2022 holdout has only 64 fixtures.** That's a small N for the spread of metrics reported here. A 1-2% swing on accuracy is within sampling noise. For a more robust estimate, consider also evaluating against EURO 2024 held out (would require code changes to support multiple holdouts).

**4. Optuna and `select.py` are dormant for national mode.** Both implemented, neither in the production training path. Static hyperparameters and a static drop list — both will go stale as the dataset grows.

**5. SHAP is computed on the home-goals model only.** The away-goals model has no explainability artefacts. Possibly the SHAPs mirror, but unverified.

**6. CV per-fold metrics are not saved.** Only the final holdout numbers persist. CV variance / fold stability cannot be inspected from current artefacts.

**7. No reliability diagrams / calibration plots.** Standard tool for assessing probabilistic models — missing here. Recommend adding a per-mode reliability plot.

**8. No per-stage breakdown (group vs knockout).** Holdout metrics are reported as single summary numbers. Knowing the model is good at WC group stage but bad at WC knockouts would be actionable.

**9. No confusion matrix in standard flow.** `evaluate.py` defines `get_confusion_matrix` (line 197–204) but it's not called in `train_pipeline.py`.

**10. Recent-window drift is not automated.** Manual analysis (2026-04-30) showed total goals systematically under-predicted in recent club fixtures (PL −15.7%, La Liga −8.4%, Ligue 1 −26.2%). Holdout calibration is fine — this is **drift, not bias**. There's no scheduled drift-detection job; the analysis was ad-hoc. *Note: this finding is from club mode; equivalent analysis for national-mode recent fixtures hasn't been done.*

**11. Tournament running stats earn their place via theory, not data.** None of the in-tournament features appear in the top-10 SHAP. They might still matter for knockout matches specifically, but with only ~16 knockout fixtures in the WC 2022 holdout the SHAP signal is faint.

---

## 13. Reproducibility

| | |
|---|---|
| Random seeds | LightGBM uses default seed; not pinned in code. CV folds are deterministic via `TimeSeriesSplit`. |
| Data versioning | Raw API JSONs are immutable in S3 (date-prefixed). Training/inference Parquets are regenerated by the feature Lambda; no DVC / explicit versioning of derived tables. |
| Artefact versioning | `artefacts/` is overwritten on each retrain. The frozen-prediction lineage (`model_trained_at`) is the canonical "which model" pointer for a given prediction. No archived past models. |
| Environment | Lambda Docker images pin via image tag; `pyproject.toml` pins via `uv lock`. Dependencies are reproducible; the trained model itself is not byte-identical across retrains because LightGBM seed is unset. |

For a reviewer who wants a reproducible re-evaluation: the **frozen predictions** are the source of truth. Even without re-running the training pipeline, `predictions/<fid>.json` joined to actuals gives an honest accuracy view per lineage bucket.

---

## 14. Useful entry points

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

artefacts/comparison.csv            per-candidate metrics on WC 2022 holdout
artefacts/shap_feature_importance.csv
                                   mean |SHAP| per feature, sorted
outputs/shap_*.png                 summary, bar, top-5 dependence plots
outputs/training_history.csv       full training-run log with is_best flag
artefacts/{model_final_home,model_final_away,model_final_scaler}.pkl
artefacts/rho.json                 calibrated correlation parameter (= -0.106)
artefacts/shap_explainer.pkl       pickled TreeExplainer
```

CLAUDE.md and `documents/technical-architecture.md` provide system-level context. `documents/model-card-clubs.md` covers the parallel club-mode model.
