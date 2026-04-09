"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    home_team_id: int = Field(..., description="API-Football team ID for the home team")
    away_team_id: int = Field(..., description="API-Football team ID for the away team")
    league_id: int = Field(1, description="Competition league ID (default: World Cup)")
    match_date: str | None = Field(None, description="Match date (ISO format), defaults to today")


class ScorelineProbability(BaseModel):
    """Probability of a specific scoreline."""

    home_goals: int
    away_goals: int
    probability: float


class PredictResponse(BaseModel):
    """Response body for predictions."""

    home_team_id: int
    away_team_id: int
    lambda_home: float = Field(..., description="Expected goals for home team")
    lambda_away: float = Field(..., description="Expected goals for away team")
    most_likely_score: str = Field(..., description="Most probable scoreline, e.g. '1-0'")
    home_win: float = Field(..., description="P(home win)")
    draw: float = Field(..., description="P(draw)")
    away_win: float = Field(..., description="P(away win)")
    top_scorelines: list[ScorelineProbability] = Field(
        default_factory=list, description="Top 5 most likely scorelines"
    )


# ------------------------------------------------------------------
# Tournament Simulation
# ------------------------------------------------------------------


class SimulateRequest(BaseModel):
    """Request body for POST /simulate/tournament."""

    groups: dict[str, list[int]] = Field(
        ..., description="Group name → list of team IDs, e.g. {'A': [1, 2, 3, 4]}"
    )
    n_sims: int = Field(10000, description="Number of Monte Carlo simulations", ge=100, le=100000)


class TeamSimulationResult(BaseModel):
    """Per-team tournament simulation result."""

    team_id: int
    group_win_prob: float
    advance_prob: float
    qf_prob: float
    sf_prob: float
    final_prob: float
    champion_prob: float


class SimulateResponse(BaseModel):
    """Response body for tournament simulation."""

    n_sims: int
    results: list[TeamSimulationResult]


# ------------------------------------------------------------------
# Teams
# ------------------------------------------------------------------


class TeamResponse(BaseModel):
    """Response body for a single team."""

    id: int
    name: str
    country: str | None = None
    national: bool = True
    logo: str | None = None
    fifa_rank: int | None = None


class TeamListResponse(BaseModel):
    """Response body for team listing."""

    teams: list[TeamResponse]
    total: int


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    model_loaded: bool = False
    version: str = "0.1.0"
