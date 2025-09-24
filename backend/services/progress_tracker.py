"""Progress tracking service for long-running operations."""
from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from backend.services.base_service import BaseService, singleton
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ProgressStatus(str, Enum):
    """Progress task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressType(str, Enum):
    """Progress task types."""
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_PROCESSING = "document_processing"
    BATCH_PROCESSING = "batch_processing"
    COMPARISON_JOB = "comparison_job"
    COMPARISON_PAIR = "comparison_pair"
    EMBEDDING_GENERATION = "embedding_generation"


class ProgressTask:
    """Progress task representation."""

    def __init__(
        self,
        task_id: str,
        task_type: ProgressType,
        description: str,
        total_steps: Optional[int] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.status = ProgressStatus.PENDING
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_percent = 0.0
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.current_message: Optional[str] = None
        self.sub_tasks: List[str] = []
        self._listeners: List[asyncio.Queue] = []
        # 额外的进度信息
        self.items_per_second: Optional[float] = None
        self.estimated_seconds_remaining: Optional[float] = None
        self.last_update_time: datetime = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "status": self.status,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress_percent": self.progress_percent,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "current_message": self.current_message,
            "sub_tasks": self.sub_tasks,
            "duration_seconds": self._calculate_duration(),
            "items_per_second": self.items_per_second,
            "estimated_seconds_remaining": self.estimated_seconds_remaining
        }

    def _calculate_duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


@singleton
class ProgressTracker(BaseService):
    """Service for tracking progress of long-running operations."""

    def _initialize(self) -> None:
        self._tasks: Dict[str, ProgressTask] = {}
        self._task_futures: Dict[str, asyncio.Future] = {}
        self._global_listeners: List[asyncio.Queue] = []

    def create_task(
        self,
        task_type: ProgressType,
        description: str,
        total_steps: Optional[int] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new progress task."""
        self._ensure_initialized()

        task_id = str(uuid4())
        task = ProgressTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            total_steps=total_steps,
            parent_id=parent_id,
            metadata=metadata
        )

        self._tasks[task_id] = task
        self._task_futures[task_id] = asyncio.Future()

        if parent_id and parent_id in self._tasks:
            self._tasks[parent_id].sub_tasks.append(task_id)

        logger.info(
            "progress_task_created",
            task_id=task_id,
            task_type=task_type,
            description=description,
            total_steps=total_steps,
            parent_id=parent_id
        )

        asyncio.create_task(self._notify_listeners(task_id, "created"))

        return task_id

    async def start_task(self, task_id: str) -> None:
        """Mark a task as started."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = ProgressStatus.RUNNING
        task.started_at = datetime.utcnow()

        logger.info(
            "progress_task_started",
            task_id=task_id,
            task_type=task.task_type,
            description=task.description
        )

        await self._notify_listeners(task_id, "started")

    async def update_progress(
        self,
        task_id: str,
        current_step: Optional[int] = None,
        progress_percent: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """Update task progress."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]

        if current_step is not None:
            task.current_step = current_step
            if task.total_steps:
                task.progress_percent = (current_step / task.total_steps) * 100

        if progress_percent is not None:
            task.progress_percent = progress_percent

        if message:
            task.current_message = message

        # 计算速度和剩余时间
        now = datetime.utcnow()
        time_elapsed = (now - task.last_update_time).total_seconds()

        if time_elapsed > 0 and task.started_at and current_step is not None and task.current_step > 0:
            # 计算总耗时
            total_elapsed = (now - task.started_at).total_seconds()
            # 计算平均每项耗时
            if task.current_step > 0:
                task.items_per_second = task.current_step / total_elapsed
                # 估算剩余时间
                if task.total_steps and task.current_step < task.total_steps:
                    remaining_items = task.total_steps - task.current_step
                    task.estimated_seconds_remaining = remaining_items / task.items_per_second if task.items_per_second > 0 else None

        task.last_update_time = now

        logger.debug(
            "progress_task_updated",
            task_id=task_id,
            current_step=task.current_step,
            progress_percent=task.progress_percent,
            message=message,
            items_per_second=task.items_per_second,
            estimated_seconds_remaining=task.estimated_seconds_remaining
        )

        await self._notify_listeners(task_id, "progress")

    async def complete_task(self, task_id: str, message: Optional[str] = None) -> None:
        """Mark a task as completed."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = ProgressStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.progress_percent = 100.0

        if message:
            task.current_message = message

        logger.info(
            "progress_task_completed",
            task_id=task_id,
            task_type=task.task_type,
            description=task.description,
            duration_seconds=task._calculate_duration()
        )

        await self._notify_listeners(task_id, "completed")

        # Resolve the future
        if task_id in self._task_futures:
            self._task_futures[task_id].set_result(True)

    async def fail_task(self, task_id: str, error_message: str) -> None:
        """Mark a task as failed."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = ProgressStatus.FAILED
        task.completed_at = datetime.utcnow()
        task.error_message = error_message

        logger.error(
            "progress_task_failed",
            task_id=task_id,
            task_type=task.task_type,
            description=task.description,
            error_message=error_message,
            duration_seconds=task._calculate_duration()
        )

        await self._notify_listeners(task_id, "failed")

        # Resolve the future with exception
        if task_id in self._task_futures:
            self._task_futures[task_id].set_exception(Exception(error_message))

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = ProgressStatus.CANCELLED
        task.completed_at = datetime.utcnow()

        logger.info(
            "progress_task_cancelled",
            task_id=task_id,
            task_type=task.task_type,
            description=task.description
        )

        await self._notify_listeners(task_id, "cancelled")

        # Cancel the future
        if task_id in self._task_futures:
            self._task_futures[task_id].cancel()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            return None

        return self._tasks[task_id].to_dict()

    def get_tasks_by_parent(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all subtasks of a parent task."""
        self._ensure_initialized()

        return [
            task.to_dict()
            for task in self._tasks.values()
            if task.parent_id == parent_id
        ]

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active (running) tasks."""
        self._ensure_initialized()

        return [
            task.to_dict()
            for task in self._tasks.values()
            if task.status == ProgressStatus.RUNNING
        ]

    def get_recent_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent tasks sorted by creation time."""
        self._ensure_initialized()

        sorted_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )

        return [task.to_dict() for task in sorted_tasks[:limit]]

    async def wait_for_task(self, task_id: str) -> bool:
        """Wait for a task to complete."""
        self._ensure_initialized()

        if task_id not in self._task_futures:
            raise ValueError(f"Task {task_id} not found")

        return await self._task_futures[task_id]

    def subscribe_task(self, task_id: str) -> asyncio.Queue:
        """Subscribe to updates for a specific task."""
        self._ensure_initialized()

        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        queue = asyncio.Queue()
        self._tasks[task_id]._listeners.append(queue)

        return queue

    def subscribe_all(self) -> asyncio.Queue:
        """Subscribe to updates for all tasks."""
        self._ensure_initialized()

        queue = asyncio.Queue()
        self._global_listeners.append(queue)

        return queue

    def unsubscribe_task(self, task_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from task updates."""
        self._ensure_initialized()

        if task_id in self._tasks:
            task = self._tasks[task_id]
            if queue in task._listeners:
                task._listeners.remove(queue)

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from all task updates."""
        self._ensure_initialized()

        if queue in self._global_listeners:
            self._global_listeners.remove(queue)

    async def _notify_listeners(self, task_id: str, event: str) -> None:
        """Notify all listeners about task update."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        update = {
            "event": event,
            "task": task.to_dict()
        }

        # Notify task-specific listeners
        for queue in task._listeners[:]:
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                # Remove full queues
                task._listeners.remove(queue)

        # Notify global listeners
        for queue in self._global_listeners[:]:
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                # Remove full queues
                self._global_listeners.remove(queue)

    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks."""
        self._ensure_initialized()

        cutoff_time = datetime.utcnow()
        cutoff_seconds = older_than_hours * 3600

        tasks_to_remove = []
        for task_id, task in self._tasks.items():
            if task.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                if task.completed_at:
                    age = (cutoff_time - task.completed_at).total_seconds()
                    if age > cutoff_seconds:
                        tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            if task_id in self._task_futures:
                del self._task_futures[task_id]

        logger.info(
            "progress_tasks_cleaned_up",
            removed_count=len(tasks_to_remove),
            older_than_hours=older_than_hours
        )

        return len(tasks_to_remove)


# Context manager for progress tracking
class ProgressContext:
    """Context manager for automatic progress tracking."""

    def __init__(
        self,
        tracker: ProgressTracker,
        task_type: ProgressType,
        description: str,
        total_steps: Optional[int] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tracker = tracker
        self.task_type = task_type
        self.description = description
        self.total_steps = total_steps
        self.parent_id = parent_id
        self.metadata = metadata
        self.task_id: Optional[str] = None

    async def __aenter__(self) -> str:
        """Create and start the task."""
        self.task_id = self.tracker.create_task(
            task_type=self.task_type,
            description=self.description,
            total_steps=self.total_steps,
            parent_id=self.parent_id,
            metadata=self.metadata
        )
        await self.tracker.start_task(self.task_id)
        return self.task_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete or fail the task based on exception."""
        if self.task_id:
            if exc_type is None:
                await self.tracker.complete_task(self.task_id)
            else:
                await self.tracker.fail_task(
                    self.task_id,
                    f"{exc_type.__name__}: {str(exc_val)}"
                )