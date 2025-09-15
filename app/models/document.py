"""
文档数据模型 - Document相关的所有模型类
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChunkType(str, Enum):
    """文本块类型"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class DocumentBase(BaseModel):
    """文档基础模型"""
    content: str = Field(..., min_length=1, max_length=500000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpload(DocumentBase):
    """文档上传请求"""
    chunk_size: int = Field(500, ge=100, le=2000)
    overlap: int = Field(50, ge=0, le=500)


class DocumentChunk(BaseModel):
    """文档块"""
    id: str
    document_id: str
    content: str
    chunk_type: ChunkType
    position: int
    embedding: Optional[List[float]] = None


class Document(DocumentBase):
    """文档实体"""
    id: str
    created_at: datetime
    chunks_count: int
    status: str = "processing"