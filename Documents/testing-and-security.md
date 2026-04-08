# Testing & Vulnerability Scanning Guide

> Unit testing, integration testing, end-to-end testing, and security scanning for the football predictions platform.

---

## Table of Contents

1. [Testing Strategy Overview](#testing-strategy-overview)
2. [Python Unit Tests](#python-unit-tests)
3. [Angular Unit Tests](#angular-unit-tests)
4. [Node.js BFF Tests](#nodejs-bff-tests)
5. [Integration Tests](#integration-tests)
6. [End-to-End Tests](#end-to-end-tests)
7. [Vulnerability Scanning](#vulnerability-scanning)
8. [CI Pipeline Integration](#ci-pipeline-integration)

---

## Testing Strategy Overview

| Layer | Framework | What to Test |
|---|---|---|
| Python (data pipeline, ML) | pytest | Data transforms, feature engineering, model I/O, API routes |
| Angular (frontend) | Karma + Jasmine | Components, services, pipes, interceptors |
| Node.js (BFF) | Jest | Route handlers, middleware, proxy logic |
| Integration | pytest + httpx | FastAPI endpoints with test DB |
| End-to-end | Cypress or Playwright | Full user flows through Angular UI |

**Coverage target:** 80%+ on business logic (features, model, API routes). UI components can be lower but all services should be covered.

---

## Python Unit Tests

### Setup

```
tests/
├── unit/
│   ├── data/
│   │   ├── test_api_client.py
│   │   ├── test_ingest.py
│   │   └── test_schemas.py
│   ├── features/
│   │   ├── test_rolling.py
│   │   ├── test_squad.py
│   │   ├── test_h2h.py
│   │   └── test_tournament.py
│   ├── models/
│   │   ├── test_train.py
│   │   ├── test_evaluate.py
│   │   └── test_calibrate.py
│   └── api/
│       ├── test_predictions.py
│       └── test_teams.py
├── integration/
│   └── test_api_integration.py
└── fixtures/
    ├── sample_fixtures.json
    ├── sample_teams.json
    └── sample_players.json
```

### Key Testing Patterns

**Feature engineering — guard against data leakage:**

```python
def test_rolling_features_no_leakage():
    """Features for match on date D must only use data from before D."""
    df = load_test_fixtures()
    features = compute_rolling_features(df, window=10)

    for _, row in features.iterrows():
        match_date = row["date"]
        source_dates = get_source_dates_for_row(row)
        assert all(d < match_date for d in source_dates), \
            f"Leakage detected: feature uses data from {max(source_dates)} for match on {match_date}"
```

**Model I/O — verify predict contract:**

```python
def test_predict_returns_three_classes(trained_model, sample_features):
    """Prediction must return probabilities for all 3 outcome classes."""
    proba = trained_model.predict_proba(sample_features)
    assert proba.shape[1] == 3
    assert all(abs(row.sum() - 1.0) < 1e-6 for row in proba)
```

**API client — mock external calls:**

```python
@pytest.fixture
def mock_api_response():
    return {"response": [{"fixture": {"id": 1}, "teams": {...}, "goals": {...}}]}

def test_fetch_fixtures_parses_response(mock_api_response, mocker):
    mocker.patch("src.data.api_client.requests.get", return_value=MockResponse(mock_api_response))
    fixtures = fetch_fixtures(league_id=1, season=2022)
    assert len(fixtures) == 1
    assert fixtures[0]["fixture_id"] == 1
```

### Running

```bash
pytest tests/unit -v --cov=src --cov-report=term-missing
```

---

## Angular Unit Tests

### What to Test

| Target | Test Focus |
|---|---|
| Components | Rendering, input/output bindings, user interactions |
| Services | HTTP calls (mock HttpClient), data transforms, error handling |
| Pipes | Transform logic with edge cases |
| Interceptors | Header injection, error mapping |
| Guards | Route access logic |

### Example — Service Test

```typescript
describe('PredictionService', () => {
  let service: PredictionService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [PredictionService],
    });
    service = TestBed.inject(PredictionService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  it('should return match prediction with three outcome probabilities', () => {
    const mockPrediction = {
      home_win: 0.42,
      draw: 0.27,
      away_win: 0.31,
      predicted: 'home_win',
    };

    service.getPrediction('fixture-123').subscribe((result) => {
      expect(result.home_win + result.draw + result.away_win).toBeCloseTo(1.0);
      expect(result.predicted).toBe('home_win');
    });

    const req = httpMock.expectOne('/api/predictions/fixture-123');
    req.flush(mockPrediction);
  });

  afterEach(() => httpMock.verify());
});
```

### Example — Component Test

```typescript
describe('MatchCardComponent', () => {
  it('should display both team names', () => {
    const fixture = TestBed.createComponent(MatchCardComponent);
    fixture.componentInstance.match = {
      homeTeam: 'France',
      awayTeam: 'Brazil',
      prediction: { home_win: 0.5, draw: 0.25, away_win: 0.25 },
    };
    fixture.detectChanges();

    const el = fixture.nativeElement;
    expect(el.textContent).toContain('France');
    expect(el.textContent).toContain('Brazil');
  });
});
```

### Running

```bash
cd src/ui/client
ng test --watch=false --code-coverage
```

---

## Node.js BFF Tests

### What to Test

| Target | Test Focus |
|---|---|
| Route handlers | Request/response shape, status codes, proxy behaviour |
| Middleware | Error formatting, logging side effects |
| Config | Env variable parsing, defaults |

### Example — Route Test

```typescript
import request from 'supertest';
import { app } from '../src/index';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('GET /api/predictions/:fixtureId', () => {
  it('proxies to FastAPI and returns prediction', async () => {
    mockedAxios.get.mockResolvedValue({
      data: { home_win: 0.4, draw: 0.3, away_win: 0.3, predicted: 'home_win' },
    });

    const res = await request(app).get('/api/predictions/123');

    expect(res.status).toBe(200);
    expect(res.body.predicted).toBe('home_win');
    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.stringContaining('/predictions/123')
    );
  });

  it('returns 502 when FastAPI is unreachable', async () => {
    mockedAxios.get.mockRejectedValue(new Error('ECONNREFUSED'));

    const res = await request(app).get('/api/predictions/123');

    expect(res.status).toBe(502);
  });
});
```

### Running

```bash
cd src/ui/server
npx jest --coverage
```

---

## Integration Tests

Test the FastAPI endpoints with a real (test) database, verifying the full request lifecycle.

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_predict_endpoint_returns_valid_prediction(client):
    response = await client.post("/predict", json={
        "home_team_id": 2,
        "away_team_id": 6,
        "league_id": 1,
        "match_date": "2026-07-01"
    })
    assert response.status_code == 200
    body = response.json()
    assert "home_win" in body
    assert "draw" in body
    assert "away_win" in body
    assert abs(body["home_win"] + body["draw"] + body["away_win"] - 1.0) < 0.01

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## End-to-End Tests

Use Cypress or Playwright to test critical user flows through the Angular UI.

### Critical Flows to Cover

1. **Dashboard loads** — upcoming matches render with predictions
2. **Match detail** — clicking a match shows full prediction breakdown and SHAP chart
3. **Tournament view** — group tables and bracket render correctly
4. **Error states** — API failure shows user-friendly error message, not a blank screen

### Example — Cypress

```typescript
describe('Dashboard', () => {
  beforeEach(() => cy.visit('/'));

  it('displays upcoming matches with predictions', () => {
    cy.get('[data-cy="match-card"]').should('have.length.at.least', 1);
    cy.get('[data-cy="match-card"]').first().within(() => {
      cy.get('[data-cy="home-team"]').should('not.be.empty');
      cy.get('[data-cy="away-team"]').should('not.be.empty');
      cy.get('[data-cy="prediction-bar"]').should('exist');
    });
  });

  it('navigates to match detail on click', () => {
    cy.get('[data-cy="match-card"]').first().click();
    cy.url().should('include', '/match/');
    cy.get('[data-cy="shap-chart"]').should('exist');
  });
});
```

---

## Vulnerability Scanning

### 1. Dependency Scanning

Detect known vulnerabilities in third-party packages.

| Tool | Scope | Command |
|---|---|---|
| `pip-audit` | Python dependencies | `pip-audit --strict --desc` |
| `npm audit` | Node.js / Angular dependencies | `npm audit --audit-level=moderate` |
| `osv-scanner` | All ecosystems (Google OSV DB) | `osv-scanner --lockfile=requirements.txt` |

Run on every CI build. **Block merges** on critical/high severity findings.

### 2. Static Application Security Testing (SAST)

Scan source code for security anti-patterns.

| Tool | Scope | What It Catches |
|---|---|---|
| `bandit` | Python | SQL injection, hardcoded secrets, unsafe deserialization, shell injection |
| `semgrep` | Python, TypeScript, JS | OWASP Top 10 patterns, custom rules |
| `eslint-plugin-security` | Node.js / Angular | Regex DoS, eval usage, non-literal requires |

```bash
# Python SAST
bandit -r src/ -c pyproject.toml
semgrep --config=auto src/

# TypeScript/JS SAST
# Add eslint-plugin-security to .eslintrc and run as part of normal lint
```

### 3. Container Image Scanning

Scan Docker images for OS-level and library vulnerabilities before deployment.

| Tool | Notes |
|---|---|
| `trivy` | Fast, covers OS packages + language libs. Run locally and in CI. |
| AWS ECR image scanning | Built-in, runs automatically on push to ECR. |

```bash
# Scan locally before pushing
trivy image football-predictions-api:latest --severity HIGH,CRITICAL
trivy image football-predictions-bff:latest --severity HIGH,CRITICAL
```

### 4. Secrets Detection

Prevent API keys, passwords, and tokens from being committed to the repository.

| Tool | Notes |
|---|---|
| `gitleaks` | Pre-commit hook + CI scan. Catches secrets in git history. |
| `trufflehog` | Deep git history scan. Good for auditing existing repos. |

```bash
# Pre-commit hook
gitleaks protect --staged

# Full repo scan
gitleaks detect --source=. --verbose
```

**Setup as a pre-commit hook:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```

### 5. Infrastructure Scanning

Scan CDK/CloudFormation templates for misconfigurations.

| Tool | What It Catches |
|---|---|
| `checkov` | Open S3 buckets, unencrypted RDS, missing logging, overly permissive IAM |
| `cdk-nag` | AWS CDK-specific best practice checks (add as CDK aspect) |

```bash
# Synth CDK and scan the output
cdk synth > template.yaml
checkov -f template.yaml
```

```python
# In CDK app — add cdk-nag as an aspect
from cdk_nag import AwsSolutionsChecks
import aws_cdk as cdk

app = cdk.App()
cdk.Aspects.of(app).add(AwsSolutionsChecks())
```

---

## CI Pipeline Integration

All testing and scanning runs in GitHub Actions on every push and PR.

```yaml
# .github/workflows/ci.yml (simplified)
name: CI
on: [push, pull_request]

jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e ".[dev]"
      - run: ruff check src/
      - run: ruff format --check src/
      - run: mypy src/
      - run: pytest tests/unit -v --cov=src --cov-fail-under=80
      - run: bandit -r src/ -c pyproject.toml
      - run: pip-audit --strict

  angular:
    runs-on: ubuntu-latest
    defaults:
      run: { working-directory: src/ui/client }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npx ng lint
      - run: npx ng test --watch=false --code-coverage
      - run: npm audit --audit-level=moderate

  node-bff:
    runs-on: ubuntu-latest
    defaults:
      run: { working-directory: src/ui/server }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npx eslint src/
      - run: npx jest --coverage
      - run: npm audit --audit-level=moderate

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - run: pip install semgrep
      - run: semgrep --config=auto src/ --error
      - uses: zricethezav/gitleaks-action@v2
      - run: pip install checkov && cdk synth > template.yaml && checkov -f template.yaml

  docker:
    runs-on: ubuntu-latest
    needs: [python, angular, node-bff]
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t api:ci -f Dockerfile .
      - run: docker build -t bff:ci -f src/ui/server/Dockerfile .
      - uses: aquasecurity/trivy-action@master
        with: { image-ref: 'api:ci', severity: 'HIGH,CRITICAL', exit-code: '1' }
      - uses: aquasecurity/trivy-action@master
        with: { image-ref: 'bff:ci', severity: 'HIGH,CRITICAL', exit-code: '1' }
```

### Gate Rules

| Check | Merge Blocked? |
|---|---|
| Unit tests failing | Yes |
| Coverage < 80% (business logic) | Yes |
| Lint errors | Yes |
| `bandit` high/critical findings | Yes |
| `pip-audit` / `npm audit` critical CVEs | Yes |
| `gitleaks` secrets detected | Yes |
| `trivy` critical image vulnerabilities | Yes |
| `checkov` high-severity infra issues | Yes (on infra changes) |
