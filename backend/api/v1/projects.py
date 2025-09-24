"""Project management APIs for grouping document comparisons."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
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


class ProjectComparisonTaskResponse(BaseModel):
    task_id: str
    job_id: Optional[int] = None
    message: str


@router.post(
    "/{project_id}/run-comparisons",
    response_model=ProjectComparisonTaskResponse,
    summary="Run or rerun comparisons for a project",
)
async def run_project_comparisons(
    project_id: int,
    background_tasks: BackgroundTasks,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> ProjectComparisonTaskResponse:
    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check if there are enough documents
    from backend.db.models import Document, DocumentStatus
    from backend.db import get_session
    from sqlalchemy import select

    async with get_session() as session:
        stmt = (
            select(Document)
            .where(Document.project_id == project_id)
            .where(Document.status == DocumentStatus.COMPLETED)
        )
        documents = (await session.exec(stmt)).all()

    if len(documents) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two completed documents are required to run comparisons"
        )

    # Create a progress task for the comparison job
    from backend.services.progress_tracker import ProgressTracker, ProgressType
    progress_tracker = ProgressTracker()

    comparison_task_id = progress_tracker.create_task(
        task_type=ProgressType.COMPARISON_JOB,
        description=f"Running comparisons for project {project_id}",
        total_steps=1,  # Will be updated once we know the pair count
        metadata={"project_id": project_id}
    )

    # Run comparisons asynchronously
    async def run_comparisons():
        await progress_tracker.start_task(comparison_task_id)
        try:
            job = await orchestrator.run_project_comparisons(project_id)
            if job:
                # Note: job ID is already in the metadata when the task was created
                await progress_tracker.complete_task(
                    comparison_task_id,
                    message=f"Comparisons completed for project {project_id}"
                )
            else:
                await progress_tracker.fail_task(
                    comparison_task_id,
                    error_message="Failed to create comparison job"
                )
        except Exception as exc:
            logger.error("Project comparison failed", project_id=project_id, error=str(exc))
            await progress_tracker.fail_task(
                comparison_task_id,
                error_message=f"Comparison failed: {str(exc)}"
            )

    background_tasks.add_task(run_comparisons)

    return ProjectComparisonTaskResponse(
        task_id=comparison_task_id,
        message=f"Started comparison process for project {project_id}"
    )
