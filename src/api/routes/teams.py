"""Team listing and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import FeatureStore, get_feature_store
from src.api.models import TeamListResponse, TeamResponse

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("", response_model=TeamListResponse)
def list_teams(
    feature_store: FeatureStore = Depends(get_feature_store),
) -> TeamListResponse:
    """List all known national teams."""
    teams = [
        TeamResponse(id=tid, name=name)
        for name, tid in sorted(feature_store.team_lookup.items(), key=lambda x: x[0])
    ]
    return TeamListResponse(teams=teams, total=len(teams))


@router.get("/{team_id}", response_model=TeamResponse)
def get_team(
    team_id: int,
    feature_store: FeatureStore = Depends(get_feature_store),
) -> TeamResponse:
    """Get a single team by ID."""
    name = feature_store.get_team_name(team_id)
    if name is None:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return TeamResponse(id=team_id, name=name)
