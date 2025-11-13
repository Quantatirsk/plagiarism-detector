"""
通用数据模型 - 共享的枚举和基础模型
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ResponseBase(BaseModel):
    """基础响应模型"""
    success: bool
    message: str | None = None
    data: dict[str, Any] | None = None


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = 1
    size: int = 50
    total: int | None = None


class TimestampedModel(BaseModel):
    """带时间戳的基础模型"""
    created_at: datetime
    updated_at: datetime | None = None


class TaskStatus(str):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
