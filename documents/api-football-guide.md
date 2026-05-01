# API-Football — Complete Usage Guide

> **Source:** [api-football.com](https://www.api-football.com) | **API Version:** v3

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [Base URL](#base-url)
5. [Key Endpoints](#key-endpoints)
6. [National Teams](#national-teams)
7. [Code Examples](#code-examples)
8. [Plans & Limits](#plans--limits)
9. [Tips & Best Practices](#tips--best-practices)

---

## Overview

API-Football is a RESTful API providing comprehensive football (soccer) data including:

- **Livescores** — real-time match data updated every 15 seconds
- **Fixtures** — past and upcoming matches
- **Standings** — league tables
- **Teams & Players** — rosters, stats, transfers, trophies
- **Odds** — pre-match and in-play bookmaker odds
- **Predictions** — match outcome predictions
- **Events** — goals, cards, substitutions
- **Line-ups** — starting XIs and substitutes

Coverage spans **1,200+ leagues and cups** worldwide.

---

## Getting Started

1. Register for a free account (no credit card needed):
   👉 [https://dashboard.api-football.com/register](https://dashboard.api-football.com/register)

2. After registration, retrieve your **API key** from your dashboard profile:
   👉 [https://dashboard.api-football.com/profile?access](https://dashboard.api-football.com/profile?access)

3. Start making requests using the base URL and your key in the request header.

---

## Authentication

Every request must include your API key as a **request header**:

```
x-apisports-key: YOUR_API_KEY_HERE
```

There is no OAuth, no token expiry, and no token exchange — just a single header on every request.

---

## Base URL

```
https://v3.football.api-sports.io
```

All endpoints are appended to this base URL.

---

## Key Endpoints

### 🌍 Countries
```
GET /countries
```
Returns a list of all available countries.

---

### 🏆 Leagues
```
GET /leagues
GET /leagues?id=39           # Premier League
GET /leagues?country=England
GET /leagues?season=2024
```

---

### 📅 Fixtures (Matches)
```
GET /fixtures?league=39&season=2024          # All Premier League 2024 fixtures
GET /fixtures?date=2024-04-08                # Fixtures on a specific date
GET /fixtures?live=all                       # All live matches right now
GET /fixtures?id=1035015                     # Single fixture by ID
```

---

### 📊 Standings
```
GET /standings?league=39&season=2024
```

---

### 👤 Players
```
GET /players?league=39&season=2024           # Players in a league
GET /players/topscorers?league=39&season=2024  # Top scorers
GET /players/squads?team=33                  # Squad for a team
```

---

### 🏟️ Teams
```
GET /teams?id=33                             # Manchester United
GET /teams?name=Barcelona
GET /teams?league=39&season=2024
```

---

### 📋 Events (Goals, Cards, etc.)
```
GET /fixtures/events?fixture=1035015
```

---

### 🔢 Line-ups
```
GET /fixtures/lineups?fixture=1035015
```

---

### 📈 Statistics
```
GET /fixtures/statistics?fixture=1035015     # Match stats (shots, possession, etc.)
GET /players/statistics?league=39&season=2024&player=276  # Player stats
```

---

### 🎲 Odds
```
GET /odds?fixture=1035015                    # Pre-match odds
GET /odds/live?fixture=1035015               # In-play odds
```

---

### 🔮 Predictions
```
GET /predictions?fixture=1035015
```

---

### ⚽ Head-to-Head
```
GET /fixtures/headtohead?h2h=33-34           # Man Utd vs Liverpool
```

---

## National Teams

National teams are fully supported in API-Football and work exactly like club teams — the same endpoints, the same response format, the same statistics. The only difference is a `"national": true` flag in the team object that distinguishes them from club sides.

### 🔍 Identifying National Teams

When you query the `/teams` endpoint, each result includes a `national` field:

```json
{
  "team": {
    "id": 2,
    "name": "France",
    "country": "France",
    "national": true,
    "logo": "https://media.api-sports.io/football/teams/2.png"
  }
}
```

### 🌐 Finding National Teams

```
GET /teams?name=France            # Find France national team & get its ID
GET /teams?name=Brazil
GET /teams?name=Germany
```

### 🏆 Key International Competition League IDs

International tournaments are treated as leagues, each with a fixed ID:

| Competition | League ID |
|---|---|
| FIFA World Cup | 1 |
| UEFA European Championship (EURO) | 4 |
| UEFA Nations League | 5 |
| Copa América | 9 |
| Africa Cup of Nations (AFCON) | 6 |
| CONCACAF Gold Cup | 7 |
| AFC Asian Cup | 10 |
| International Friendlies | 11 |

> **Tip:** Use `GET /leagues?type=Cup&country=World` to discover all international competitions and their IDs.

### 📅 National Team Fixtures

```
# All World Cup 2022 matches
GET /fixtures?league=1&season=2022

# France's fixtures in the 2024 EURO
GET /fixtures?league=4&season=2024&team=2

# Live international matches right now
GET /fixtures?live=all&league=5
```

### 📊 National Team Standings / Group Tables

```
GET /standings?league=1&season=2022    # World Cup 2022 group tables
GET /standings?league=4&season=2024    # EURO 2024 group tables
GET /standings?league=5&season=2024    # UEFA Nations League standings
```

### 👤 Player Stats in International Competitions

Player statistics are scoped per competition and season, so international stats are separate from club stats:

```
# Top scorers at World Cup 2022
GET /players/topscorers?league=1&season=2022

# France squad at World Cup 2022
GET /players/squads?team=2

# Stats for a specific player in a tournament
GET /players?id=276&league=1&season=2022
```

### 📋 Match Events, Line-ups & Statistics

All fixture-level endpoints work identically for international matches — just use the fixture ID from a national team game:

```
GET /fixtures/events?fixture=855748       # Goals, cards, subs in a World Cup match
GET /fixtures/lineups?fixture=855748      # Starting XIs
GET /fixtures/statistics?fixture=855748  # Possession, shots, passes, etc.
```

### ⚽ Head-to-Head Between National Teams

```
# France vs Brazil all-time history
GET /fixtures/headtohead?h2h=2-6
```

### ⚠️ Key Differences vs. Club Teams

- **Stats are tournament-scoped** — a player's World Cup stats are separate from their club league stats. Always specify `league` and `season` when querying player stats.
- **No transfers or trophies endpoints** — these are club-specific features not applicable to national teams.
- **Squad depth varies** — not all national teams have complete squad/player data for every competition. Check coverage first.
- **Standings format differs** — international group stages return groups (A, B, C…) rather than a single ranked table.

---

## Code Examples

### cURL
```bash
curl --request GET \
  --url 'https://v3.football.api-sports.io/fixtures?live=all' \
  --header 'x-apisports-key: YOUR_API_KEY_HERE'
```

---

### JavaScript (fetch)
```javascript
const response = await fetch(
  'https://v3.football.api-sports.io/standings?league=39&season=2024',
  {
    headers: {
      'x-apisports-key': 'YOUR_API_KEY_HERE'
    }
  }
);

const data = await response.json();
console.log(data.response); // Array of standings
```

---

### Python (requests)
```python
import requests

url = "https://v3.football.api-sports.io/players/topscorers"
params = {"league": 39, "season": 2024}
headers = {"x-apisports-key": "YOUR_API_KEY_HERE"}

response = requests.get(url, headers=headers, params=params)
data = response.json()

for player in data["response"]:
    name = player["player"]["name"]
    goals = player["statistics"][0]["goals"]["total"]
    print(f"{name}: {goals} goals")
```

---

### Node.js (axios)
```javascript
const axios = require('axios');

const { data } = await axios.get(
  'https://v3.football.api-sports.io/fixtures',
  {
    params: { league: 39, season: 2024, date: '2024-04-08' },
    headers: { 'x-apisports-key': 'YOUR_API_KEY_HERE' }
  }
);

data.response.forEach(fixture => {
  console.log(`${fixture.teams.home.name} vs ${fixture.teams.away.name}`);
});
```

---

## Response Format

All responses follow this structure:

```json
{
  "get": "fixtures",
  "parameters": { "league": "39", "season": "2024" },
  "errors": [],
  "results": 38,
  "paging": { "current": 1, "total": 1 },
  "response": [
    { ... }
  ]
}
```

- **`response`** — the array of actual data objects
- **`results`** — total number of records returned
- **`errors`** — any API-level errors (empty array if none)

---

## Plans & Limits

| Plan  | Price    | Requests/Day |
|-------|----------|-------------|
| Free  | $0       | 100         |
| Pro   | $19/mo   | 7,500       |
| Ultra | $29/mo   | 75,000      |
| Mega  | $39/mo   | 150,000     |

- The **Free plan** gives access to all endpoints (limited historical seasons).
- No credit card required for the free tier.
- A single account also unlocks free access to 11 other API-Sports products (Basketball, NBA, NFL, F1, etc.).

---

## Tips & Best Practices

1. **Get IDs first** — Most endpoints require IDs (league ID, team ID, fixture ID). Use the `/leagues` and `/teams` endpoints to look them up before building other queries.

2. **Endpoint dependency chain:**
   ```
   League ID → Fixture IDs → Events / Line-ups / Statistics
   Player ID  → Player Stats / Career Profile
   ```

3. **Cache responses** — With only 100 requests/day on the free tier, cache data locally (especially standings and fixtures) to avoid burning your quota.

4. **Use the Live Tester** — The dashboard has a built-in live tester to validate responses before writing code.

5. **Livescore polling** — Live data updates every 15 seconds, but don't poll more frequently than that or you'll waste requests.

6. **Check coverage** — Before building features for a specific league, verify it's covered with the `/leagues` endpoint and check the `coverage` flags in the response.

7. **Handle `errors` field** — Always check `data.errors` in the response — the HTTP status may be 200 even when there's a logical error (e.g., invalid parameter).

---

## Useful Links

| Resource | URL |
|----------|-----|
| Official Docs | https://www.api-football.com/documentation-v3 |
| Dashboard | https://dashboard.api-football.com |
| Coverage List | https://www.api-football.com/coverage |
| Pricing | https://www.api-football.com/pricing |
| Widgets | https://www.api-football.com/widgets |
