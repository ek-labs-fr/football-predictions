"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import ModelStore, get_model_store
from src.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(model_store: ModelStore = Depends(get_model_store)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_store.loaded,
    )
