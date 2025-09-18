"""
通用数据模型 - 共享的枚举和基础模型
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class ResponseBase(BaseModel):
    """基础响应模型"""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = 1
    size: int = 50
    total: Optional[int] = None


class TimestampedModel(BaseModel):
    """带时间戳的基础模型"""
    created_at: datetime
    updated_at: Optional[datetime] = None


class TaskStatus(str):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"