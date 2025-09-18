"""Project management APIs for grouping document comparisons."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from backend.db.models import CompareJob, CompareJobStatus, Project
from backend.services.detection_orchestrator import DetectionOrchestrator

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])


def _orchestrator() -> DetectionOrchestrator:
    return DetectionOrchestrator()


class ProjectCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1024)


class ProjectResponse(BaseModel):
    id: int
    name: Optional[str]
    description: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, project: Project) -> "ProjectResponse":
        return cls(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
        )


class ProjectListResponse(BaseModel):
    items: List[ProjectResponse]


class ProjectJobResponse(BaseModel):
    id: int
    project_id: int
    name: Optional[str]
    status: CompareJobStatus
    created_at: Optional[str]
    updated_at: Optional[str]
    config: Optional[dict]

    @classmethod
    def from_model(cls, job: CompareJob) -> "ProjectJobResponse":
        return cls(
            id=job.id,
            project_id=job.project_id,
            name=job.name,
            status=job.status,
            created_at=job.created_at.isoformat() if job.created_at else None,
            updated_at=job.updated_at.isoformat() if job.updated_at else None,
            config=job.config_json,
        )


class ProjectJobListResponse(BaseModel):
    items: List[ProjectJobResponse]


@router.post("", response_model=ProjectResponse, summary="Create a project")
async def create_project(
    payload: ProjectCreateRequest,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectResponse:
    project = await orchestrator.create_project(payload.name, payload.description)
    return ProjectResponse.from_model(project)


@router.get("", response_model=ProjectListResponse, summary="List projects")
async def list_projects(
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectListResponse:
    projects = await orchestrator.list_projects()
    return ProjectListResponse(items=[ProjectResponse.from_model(project) for project in projects])


@router.get("/{project_id}", response_model=ProjectResponse, summary="Get project detail")
async def get_project(
    project_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectResponse:
    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectResponse.from_model(project)


@router.get("/{project_id}/jobs", response_model=ProjectJobListResponse, summary="List jobs for project")
async def list_project_jobs(
    project_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectJobListResponse:
    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    jobs = await orchestrator.list_compare_jobs(project_id=project_id)
    return ProjectJobListResponse(items=[ProjectJobResponse.from_model(job) for job in jobs])


@router.post(
    "/{project_id}/run-comparisons",
    response_model=ProjectJobResponse,
    summary="Run or rerun comparisons for a project",
)
async def run_project_comparisons(
    project_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectJobResponse:
    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = await orchestrator.run_project_comparisons(project_id)
    if not job:
        raise HTTPException(status_code=400, detail="At least two completed documents are required to run comparisons")

    return ProjectJobResponse.from_model(job)
