"""Comparison job APIs for managing pairwise detection."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from sqlalchemy.exc import NoResultFound
from backend.db.models import ChunkGranularity, CompareJobStatus, ComparePairStatus
from backend.services.comparison_service import ComparisonService, ComparisonConfig
from backend.services.detection_orchestrator import DetectionOrchestrator
from backend.services.similarity_pipeline import PipelineConfig

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/compare-jobs", tags=["Comparison"])


def _orchestrator() -> DetectionOrchestrator:
    return DetectionOrchestrator()


def _comparison_service() -> ComparisonService:
    return ComparisonService()


class JobCreateRequest(BaseModel):
    project_id: int
    name: Optional[str] = None
    config: Optional[dict] = None


class JobResponse(BaseModel):
    id: int
    project_id: int
    name: Optional[str]
    status: CompareJobStatus
    created_at: Optional[str]
    updated_at: Optional[str]
    config: Optional[dict]

    @classmethod
    def from_model(cls, job):
        return cls(
            id=job.id,
            project_id=job.project_id,
            name=job.name,
            status=job.status,
            created_at=job.created_at.isoformat() if job.created_at else None,
            updated_at=job.updated_at.isoformat() if job.updated_at else None,
            config=job.config_json,
        )


class JobListResponse(BaseModel):
    items: List[JobResponse]


@router.post("", response_model=JobResponse, summary="Create a comparison job")
async def create_job(
    payload: JobCreateRequest,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> JobResponse:
    project = await orchestrator.fetch_project(payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    job = await orchestrator.create_compare_job(payload.project_id, payload.name, payload.config)
    return JobResponse.from_model(job)


@router.get("", response_model=JobListResponse, summary="List comparison jobs")
async def list_jobs(
    project_id: Optional[int] = Query(None, description="Filter by project"),
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> JobListResponse:
    jobs = await orchestrator.list_compare_jobs(project_id=project_id)
    return JobListResponse(items=[JobResponse.from_model(job) for job in jobs])


class PairSpec(BaseModel):
    left_document_id: int
    right_document_id: int


class PipelineOptions(BaseModel):
    lexical_shingle_size: int = 8
    lexical_threshold: float = 0.4
    semantic_threshold: float = 0.70
    final_threshold: float = 0.75
    top_k: int = 5
    max_candidates: int = 500
    cross_encoder_top_k: int = 200
    cross_encoder_threshold: float = 0.55


class PairCreateRequest(BaseModel):
    pairs: List[PairSpec] = Field(..., min_items=1)
    execute: bool = Field(default=True, description="Run comparison immediately")
    pipeline: Optional[PipelineOptions] = None
    granularity: ChunkGranularity = Field(
        default=ChunkGranularity.PARAGRAPH,
        description="Chunk granularity for comparison"
    )


class PairResponse(BaseModel):
    id: int
    job_id: int
    left_document_id: int
    right_document_id: int
    status: ComparePairStatus
    metrics: Optional[dict]

    @classmethod
    def from_model(cls, pair):
        return cls(
            id=pair.id,
            job_id=pair.job_id,
            left_document_id=pair.left_document_id,
            right_document_id=pair.right_document_id,
            status=pair.status,
            metrics=pair.metrics_json,
        )


class PairListResponse(BaseModel):
    items: List[PairResponse]


@router.post("/{job_id}/pairs", response_model=PairListResponse, summary="Create comparison pairs")
async def create_pairs(
    job_id: int,
    payload: PairCreateRequest,
    background_tasks: BackgroundTasks,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
    comparison_service: ComparisonService = Depends(_comparison_service),
) -> PairListResponse:
    job = await orchestrator.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Comparison job not found")

    try:
        job = await orchestrator.update_compare_job_status(job_id, status=CompareJobStatus.QUEUED)
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Comparison job not found")

    tuples = [(spec.left_document_id, spec.right_document_id) for spec in payload.pairs]
    pairs = await orchestrator.add_pairs_to_job(job_id, tuples)

    if payload.execute:
        pipeline_config = PipelineConfig(**payload.pipeline.dict()) if payload.pipeline else PipelineConfig()
        config = ComparisonConfig(
            pipeline=pipeline_config,
            granularity=payload.granularity,
        )
        await orchestrator.update_compare_job_status(job_id, status=CompareJobStatus.RUNNING)

        async def _execute_pairs(job_id: int, pair_ids: list[int], config: ComparisonConfig) -> None:
            try:
                for pair_id in pair_ids:
                    await comparison_service.run_pair(pair_id, config)
            except Exception as exc:  # pragma: no cover - logged on failure
                await orchestrator.update_compare_job_status(job_id, status=CompareJobStatus.FAILED)
                logger.error("Comparison execution failed", job_id=job_id, error=str(exc))
            else:
                await orchestrator.update_compare_job_status(job_id, status=CompareJobStatus.COMPLETED)

        pair_ids = [pair.id for pair in pairs]
        background_tasks.add_task(_execute_pairs, job_id, pair_ids, config)

    return PairListResponse(items=[PairResponse.from_model(pair) for pair in pairs])


@router.delete("/{job_id}", status_code=204, summary="Delete comparison job")
async def delete_job(
    job_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> Response:
    job = await orchestrator.fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Comparison job not found")
    await orchestrator.delete_job(job_id)
    return Response(status_code=204)


@router.get("/{job_id}/pairs", response_model=PairListResponse, summary="List pairs for a job")
async def list_pairs(
    job_id: int = Path(..., description="Comparison job ID"),
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> PairListResponse:
    pairs = await orchestrator.list_pairs_for_job(job_id)
    return PairListResponse(items=[PairResponse.from_model(pair) for pair in pairs])


@router.delete("/pairs/{pair_id}", status_code=204, summary="Delete comparison pair")
async def delete_pair(
    pair_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> Response:
    pair = await orchestrator.fetch_pair(pair_id)
    if not pair:
        raise HTTPException(status_code=404, detail="Comparison pair not found")
    await orchestrator.delete_pairs([pair_id])
    return Response(status_code=204)


class MatchGroup(BaseModel):
    id: int
    left_chunk_id: int
    right_chunk_id: int
    final_score: Optional[float]
    semantic_score: Optional[float]
    lexical_overlap: Optional[float]
    cross_score: Optional[float]
    alignment_ratio: Optional[float]
    span_count: int
    match_count: int
    paragraph_spans: Optional[List[dict]]
    document_spans: Optional[List[dict]]


class MatchDetailModel(BaseModel):
    group_id: int
    left_chunk_id: int
    right_chunk_id: int
    final_score: Optional[float]
    semantic_score: Optional[float]
    lexical_overlap: Optional[float]
    cross_score: Optional[float]
    spans: Optional[List[dict]]


class PairReportResponse(BaseModel):
    pair: PairResponse
    left_document_id: int
    right_document_id: int
    groups: List[MatchGroup]
    details: List[MatchDetailModel]


@router.get("/pairs/{pair_id}", response_model=PairReportResponse, summary="Get comparison report for pair")
async def get_pair_report(
    pair_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> PairReportResponse:
    report = await orchestrator.fetch_compare_report(pair_id)
    pair = PairResponse.from_model(report.pair)
    groups = [
        MatchGroup(
            id=group.id,
            left_chunk_id=group.left_chunk_id,
            right_chunk_id=group.right_chunk_id,
            final_score=group.final_score,
            semantic_score=group.semantic_score,
            lexical_overlap=group.lexical_overlap,
            cross_score=group.cross_score,
            alignment_ratio=group.alignment_ratio,
            span_count=group.span_count,
            match_count=group.match_count,
            paragraph_spans=group.paragraph_spans_json,
            document_spans=group.document_spans_json,
        )
        for group in report.groups
    ]

    details = [
        MatchDetailModel(
            group_id=detail.group_id,
            left_chunk_id=detail.left_chunk_id,
            right_chunk_id=detail.right_chunk_id,
            final_score=detail.final_score,
            semantic_score=detail.semantic_score,
            lexical_overlap=detail.lexical_overlap,
            cross_score=detail.cross_score,
            spans=detail.spans_json,
        )
        for detail in report.details
    ]

    return PairReportResponse(
        pair=pair,
        left_document_id=report.left_document.id,
        right_document_id=report.right_document.id,
        groups=groups,
        details=details,
    )
