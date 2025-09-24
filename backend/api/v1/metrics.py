"""Metrics API for monitoring pipeline performance."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.services.pipeline_metrics import metrics_collector
from backend.services.cross_encoder_service import CrossEncoderService
from backend.services.minhash_filter import MinHashFilterStage

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    pipeline_metrics: Optional[Dict[str, Any]]
    cross_encoder_stats: Optional[Dict[str, Any]]
    minhash_stats: Optional[Dict[str, Any]]


@router.get("", response_model=MetricsResponse, summary="Get pipeline metrics")
async def get_metrics() -> MetricsResponse:
    """Get current pipeline performance metrics."""

    # Get pipeline metrics
    pipeline_metrics = metrics_collector.get_aggregated_stats()

    # Get cross-encoder cache stats
    try:
        ce_service = CrossEncoderService()
        cross_encoder_stats = ce_service.get_cache_stats()
    except Exception:
        cross_encoder_stats = None

    # Get MinHash stats
    try:
        mh_service = MinHashFilterStage()
        minhash_stats = mh_service.get_stats()
    except Exception:
        minhash_stats = None

    return MetricsResponse(
        pipeline_metrics=pipeline_metrics,
        cross_encoder_stats=cross_encoder_stats,
        minhash_stats=minhash_stats,
    )


@router.get("/current", summary="Get current pipeline metrics")
async def get_current_metrics():
    """Get metrics for the currently running pipeline (if any)."""

    if metrics_collector.current_metrics:
        return {
            "status": "running",
            "metrics": metrics_collector.current_metrics.to_dict()
        }

    return {
        "status": "idle",
        "metrics": None
    }


@router.get("/history", summary="Get historical pipeline metrics")
async def get_metrics_history(
    limit: int = 10
):
    """Get historical metrics for recent pipeline executions."""

    history = metrics_collector.metrics_history[-limit:]

    return {
        "total_executions": len(metrics_collector.metrics_history),
        "returned": len(history),
        "history": [m.to_dict() for m in history]
    }