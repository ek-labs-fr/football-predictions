# Football Predictions National — World Cup Match Outcome Predictor

## Project Overview

End-to-end production system for predicting national team football match outcomes (Win, Draw, Loss) with calibrated probabilities. Primary target: **FIFA World Cup 2026** (June–July 2026, USA/Canada/Mexico).

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | ML ecosystem, API clients, AWS SDK |
| Data source | API-Football v3 | 1,200+ leagues, full national team coverage |
| Data storage | PostgreSQL (RDS) | Relational structure fits fixtures/teams/players schema |
| Cache/raw storage | S3 | Raw API responses, model artefacts, SHAP outputs |
| ML framework | scikit-learn, XGBoost, LightGBM | Tabular ML standard; Optuna for tuning |
| Explainability | SHAP | Feature attribution for every prediction |
| Orchestration | AWS Step Functions or Airflow on ECS | Pipeline scheduling, data refresh |
| API backend | FastAPI | Async, auto-docs, Pydantic validation |
| UI backend | Node.js 20 LTS + Express | BFF layer between Angular and FastAPI |
| UI framework | Angular 18+ TypeScript | Component-based, strong typing, built-in DI and routing |
| Infrastructure | AWS CDK (Python) | IaC for all AWS resources |
| CI/CD | GitHub Actions | Lint, test, scan, deploy on push |
| Containerisation | Docker | Consistent dev/prod environments |

---

## Project Structure

```
football-predictions-national/
├── CLAUDE.md
├── Documents/                  # Reference docs (see list below)
│
├── src/
│   ├── data/                   # Data pipeline
│   │   ├── api_client.py       # API-Football client with rate limiting and caching
│   │   ├── ingest.py           # Pull fixtures, teams, players, events, H2H
│   │   ├── storage.py          # Save raw responses to S3 / local disk
│   │   └── schemas.py          # Pydantic models for API responses
│   │
│   ├── features/               # Feature engineering
│   │   ├── build.py            # Assemble flat training rows from raw data
│   │   ├── rolling.py          # Rolling averages, form strings, pre-match stats
│   │   ├── squad.py            # Squad quality aggregates
│   │   ├── h2h.py              # Head-to-head feature computation
│   │   └── tournament.py       # In-tournament running features
│   │
│   ├── models/                 # Model training and evaluation
│   │   ├── train.py            # Train all candidate models
│   │   ├── evaluate.py         # CV loop, metrics, comparison table
│   │   ├── tune.py             # Optuna hyperparameter search
│   │   ├── calibrate.py        # Post-hoc probability calibration
│   │   ├── explain.py          # SHAP value computation and plots
│   │   └── select.py           # Feature selection pipeline
│   │
│   ├── api/                    # Prediction API (FastAPI)
│   │   ├── main.py             # App entrypoint, CORS, lifespan
│   │   ├── routes/
│   │   │   ├── predictions.py  # POST /predict, GET /predictions/{fixture_id}
│   │   │   ├── teams.py        # GET /teams, GET /teams/{id}
│   │   │   └── health.py       # GET /health
│   │   ├── models.py           # Request/response Pydantic schemas
│   │   └── dependencies.py     # Model loader, feature store connection
│   │
│   └── ui/                     # Angular frontend + Node.js BFF
│       ├── client/             # Angular application (see ui-guide.md)
│       └── server/             # Node.js Express BFF (see ui-guide.md)
│
├── infrastructure/             # AWS CDK stacks (see aws-architecture.md)
├── artefacts/                  # Trained model files (gitignored, stored in S3)
├── outputs/                    # Evaluation plots and reports
├── tests/                      # Unit, integration, e2e (see testing-and-security.md)
├── scripts/                    # One-off utilities (bootstrap_data.py, backfill_features.py)
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/
```

---

## Development Conventions

### Python

- **Format:** `ruff format` (line length 100, runs in pre-commit)
- **Lint:** `ruff check` — rules defined in `pyproject.toml`
- **Types:** required on all public signatures; use `X | Y` union syntax; `-> None` required
- **Tests:** `pytest --cov=src --cov-fail-under=80`; files named `test_*.py` under `tests/`
- **Dependencies:** `uv add <pkg>` — never edit `requirements.txt` directly

### Angular (Client)

- **CLI:** Angular CLI (`ng`) for scaffolding, building, and testing
- **Formatter:** Prettier (default config)
- **Linter:** ESLint via `angular-eslint`
- **Styling:** SCSS with Angular Material or PrimeNG
- **State:** RxJS services + Angular signals for local component state
- **HTTP:** Built-in `HttpClient` with interceptors for error handling
- **Routing:** Angular Router with lazy-loaded feature modules
- **Tests:** Karma + Jasmine (unit), Cypress or Playwright (e2e)

### Node.js (BFF Server)

- **Runtime:** Node.js 20 LTS
- **Framework:** Express with TypeScript
- **Linter:** ESLint with `@typescript-eslint`
- **Tests:** Jest

### Git

- Branch naming: `feature/<name>`, `fix/<name>`, `data/<name>`
- Commit messages: imperative mood, concise
- Never commit: API keys, `.env` files, model artefact binaries, raw API response dumps
- `.gitignore` must exclude: `artefacts/`, `data/raw/`, `.env`, `node_modules/`, `__pycache__/`

### General

- Secrets in env vars or AWS Secrets Manager — never in code
- All API-Football requests go through `api_client.py` (rate limiting, retries, caching)
- Log all data pipeline runs with timestamps and row counts
- Pin all dependency versions in lock files

---

## Development Phases

1. **Data Pipeline** — API client, historical fixtures (1990–present), raw storage, fixtures DB
2. **Feature Engineering** — rolling stats, squad aggregates, H2H, tournament features, leakage validation
3. **Model Training** — candidate models, time-series CV, Optuna tuning, calibration, SHAP
4. **Prediction API** — FastAPI serving predictions with model + feature store
5. **UI** — Angular frontend + Node.js BFF with match cards, SHAP charts, tournament view
6. **AWS Deployment** — CDK stacks, CI/CD, monitoring
7. **Tournament Mode** — daily data refresh, live predictions, accuracy tracking during WC 2026

---

## Benchmark Targets

| Metric | Naive Baseline | Good | Excellent |
|---|---|---|---|
| Accuracy | ~45% | ~52% | ~57% |
| Log Loss | ~1.05 | ~0.95 | ~0.88 |
| Brier Score | ~0.24 | ~0.21 | ~0.19 |

---

## Reference Documents

| Document | Contents |
|---|---|
| `Documents/api-football-guide.md` | API-Football v3 endpoint reference, auth, national teams, rate limits |
| `Documents/worldcup-ml-data-pipeline.md` | Data pull sequence, feature list, flat row schema, rate limit strategy |
| `Documents/worldcup-ml-models-evaluation.md` | Model candidates, evaluation strategy, SHAP explainability, feature selection |
| `Documents/aws-architecture.md` | AWS services, CDK stacks, deployment flow, cost management |
| `Documents/ui-guide.md` | Angular components, services, Node.js BFF, page routes, design principles |
| `Documents/testing-and-security.md` | Unit/integration/e2e testing, dependency scanning, SAST, container scanning, secrets detection, CI gates |
| `Documents/action-plan.md` | Step-by-step implementation plan for data ingestion, feature engineering, and model development |
