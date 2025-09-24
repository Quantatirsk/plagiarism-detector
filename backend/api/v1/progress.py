"""Progress tracking API endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.core.logging import get_logger
from backend.services.progress_tracker import ProgressTracker, ProgressType, ProgressStatus

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/progress", tags=["Progress"])


def _progress_tracker() -> ProgressTracker:
    return ProgressTracker()


class ProgressTaskResponse(BaseModel):
    """Progress task response model."""
    task_id: str
    task_type: ProgressType
    description: str
    status: ProgressStatus
    total_steps: Optional[int]
    current_step: int
    progress_percent: float
    parent_id: Optional[str]
    metadata: dict
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    current_message: Optional[str]
    sub_tasks: List[str]
    duration_seconds: Optional[float]
    items_per_second: Optional[float]
    estimated_seconds_remaining: Optional[float]


@router.get("/tasks/{task_id}", response_model=ProgressTaskResponse)
async def get_task(
    task_id: str,
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> ProgressTaskResponse:
    """Get progress information for a specific task."""
    task = tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return ProgressTaskResponse(**task)


@router.get("/tasks/{task_id}/subtasks", response_model=List[ProgressTaskResponse])
async def get_subtasks(
    task_id: str,
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> List[ProgressTaskResponse]:
    """Get all subtasks of a parent task."""
    subtasks = tracker.get_tasks_by_parent(task_id)
    return [ProgressTaskResponse(**task) for task in subtasks]


@router.get("/active", response_model=List[ProgressTaskResponse])
async def get_active_tasks(
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> List[ProgressTaskResponse]:
    """Get all currently active (running) tasks."""
    tasks = tracker.get_active_tasks()
    return [ProgressTaskResponse(**task) for task in tasks]


@router.get("/recent", response_model=List[ProgressTaskResponse])
async def get_recent_tasks(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of tasks to return"),
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> List[ProgressTaskResponse]:
    """Get recent tasks sorted by creation time."""
    tasks = tracker.get_recent_tasks(limit=limit)
    return [ProgressTaskResponse(**task) for task in tasks]


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> dict:
    """Cancel a running task."""
    try:
        await tracker.cancel_task(task_id)
        return {"status": "cancelled", "task_id": task_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_tasks(
    older_than_hours: int = Query(default=24, ge=1, le=168, description="Remove tasks older than this many hours"),
    tracker: ProgressTracker = Depends(_progress_tracker)
) -> dict:
    """Clean up old completed tasks."""
    removed_count = tracker.cleanup_completed_tasks(older_than_hours=older_than_hours)
    return {
        "status": "completed",
        "removed_count": removed_count,
        "older_than_hours": older_than_hours
    }


# SSE endpoint for real-time progress updates
@router.get("/stream")
async def stream_progress(
    task_id: Optional[str] = Query(None, description="Specific task ID to monitor, or all tasks if not provided"),
    tracker: ProgressTracker = Depends(_progress_tracker)
):
    """
    Stream progress updates using Server-Sent Events (SSE).

    Connect to this endpoint to receive real-time progress updates.
    If task_id is provided, only updates for that task will be sent.
    Otherwise, all task updates will be streamed.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    import json

    async def event_generator():
        # Subscribe to updates
        if task_id:
            try:
                queue = tracker.subscribe_task(task_id)
            except ValueError:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                return
        else:
            queue = tracker.subscribe_all()

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'task_id': task_id})}\n\n"

            # Stream updates
            while True:
                try:
                    # Wait for update with timeout
                    update = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            # Unsubscribe
            if task_id:
                tracker.unsubscribe_task(task_id, queue)
            else:
                tracker.unsubscribe_all(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )