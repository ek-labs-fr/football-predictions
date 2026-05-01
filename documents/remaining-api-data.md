# Remaining API-Football Data to Ingest

> Data to pull tomorrow to complete the dataset and improve model accuracy.
> Estimated total: ~4,500 API calls (within the 7,500/day Pro limit).

---

## 1. Match Events (Priority: Low)

**Status:** Partially pulled — hit daily limit mid-pull. ~3,639 fixtures remaining.

**Endpoint:** `GET /fixtures/events?fixture={fixture_id}`

**What it provides:** Goals, yellow/red cards, substitutions, penalties per match.

**Used for:** Tournament features (accumulated cards, extra time/penalty indicators).

**Estimated API calls:** ~3,639

**Command to run:**
```bash
uv run python scripts/bootstrap_data.py --skip-players
```

**Impact:** Low — the model already achieves 0.875 MAE without event data. Cards add marginal signal.

---

## 2. Betting Odds (Priority: High)

**Status:** Not yet pulled.

**Endpoint:** `GET /odds?fixture={fixture_id}`

**What it provides:** Pre-match bookmaker odds (1X2, over/under, Asian handicap) from multiple bookmakers.

**Used for:** Implied probabilities as features — bookmakers already factor in injuries, form, motivation, and everything else. Single biggest potential accuracy boost.

**Estimated API calls:** ~3,670 (one per fixture)

**Integration plan:**
1. Add `fetch_odds(fixture_id)` to `src/data/ingest.py`
2. Extract the average 1X2 odds across bookmakers
3. Convert to implied probabilities: `P = 1/odds`, then normalise to sum to 1
4. Add columns: `odds_home_win`, `odds_draw`, `odds_away_win` to training table

**Data schema from API:**
```json
{
  "fixture": {"id": 855748},
  "bookmakers": [
    {
      "name": "Bet365",
      "bets": [
        {
          "name": "Match Winner",
          "values": [
            {"value": "Home", "odd": "2.10"},
            {"value": "Draw", "odd": "3.40"},
            {"value": "Away", "odd": "3.50"}
          ]
        }
      ]
    }
  ]
}
```

**Note:** Odds are only available for recent fixtures. Pre-2015 coverage is likely sparse.

---

## 3. Match Statistics (Priority: Medium)

**Status:** Not yet pulled.

**Endpoint:** `GET /fixtures/statistics?fixture={fixture_id}`

**What it provides:** Per-match team stats — shots on/off target, possession, corners, fouls, passes, expected goals (xG).

**Used for:** Enriching rolling form features with quality indicators beyond just goals. Shot accuracy and xG are stronger predictors than goals alone.

**Estimated API calls:** ~3,670 (one per fixture, but could batch with events)

**Integration plan:**
1. Add `fetch_match_statistics(fixture_id)` to `src/data/ingest.py`
2. Extract key stats per team: `shots_on`, `shots_off`, `possession`, `corners`, `fouls`, `passes_accurate`, `xg`
3. Compute rolling averages in `src/features/rolling.py`: `xg_avg_l10`, `shots_on_avg_l10`, `possession_avg_l10`
4. Add as features to training table

**Note:** xG data may only be available for recent seasons (2020+). Older matches may lack this.

---

## 4. Injuries / Suspensions (Priority: Medium-High)

**Status:** Not yet pulled.

**Endpoint:** `GET /injuries?fixture={fixture_id}` or `GET /injuries?team={team_id}&season={season}`

**What it provides:** Injured/suspended players per team before a match — player name, type (injury/suspension), reason.

**Used for:** A feature like `key_players_missing` (count of injured starters) or `total_minutes_missing` (sum of missing players' typical match minutes). Missing a star player is one of the biggest factors fans use that our model currently ignores.

**Estimated API calls:** ~3,670 (per fixture) or ~1,500 (per team/season combo)

**Integration plan:**
1. Add `fetch_injuries(fixture_id)` to `src/data/ingest.py`
2. Cross-reference with player ratings from `players.csv`
3. Compute: `home_injuries_count`, `home_injuries_quality` (sum of ratings of missing players), same for away
4. Add to training table

**Note:** Injury data coverage varies — may only be available for major tournaments and recent qualifiers.

---

## Recommended Pull Order

Given the 7,500/day limit:

| Order | Data | Calls | Running Total |
|---|---|---|---|
| 1 | Match events (finish) | ~3,639 | 3,639 |
| 2 | Betting odds | ~3,670 | 7,309 |

This fits within one day. Match statistics and injuries would need a second day:

| Order | Data | Calls | Running Total |
|---|---|---|---|
| 3 | Match statistics | ~3,670 | 3,670 |
| 4 | Injuries | ~1,500 | 5,170 |

### Day 1 Command
```bash
# Finish events, then pull odds
uv run python scripts/bootstrap_data.py --skip-players
# Then run a separate odds pull script (to be built)
```

### Day 2 Command
```bash
# Pull match statistics and injuries (scripts to be built)
```

---

## Data Not Available from API-Football

These would need alternative sources:

| Data | Why it matters | Where to get it |
|---|---|---|
| **Elo ratings** | Better strength metric than FIFA rank | ✅ Already ingested (Kaggle, 1872–2025) |
| **FIFA rankings** | Used in current model | ✅ Already ingested (1992–2026) |
| **Venue altitude** | High-altitude advantage (Bolivia, Ecuador, Colombia) | Manual CSV |
| **Travel distance** | Jet lag / travel fatigue for away teams | Geocoding API |
| **Manager tenure** | New manager bounce / stability | Manual research |
| **Political/war factors** | Teams affected by conflicts | Manual flags |
