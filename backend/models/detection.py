"""
检测数据模型 - 简化版：只保留双文档对比相关模型
"""
from pydantic import BaseModel
from typing import List, Optional


class SimilarityMatch(BaseModel):
    """相似度匹配结果"""
    query_text: str
    matched_text: str
    similarity_score: float
    document_id: str  # 匹配到的文档ID（目标文档）
    query_document_id: str  # 查询来源文档ID
    position: int
    # 便于前端稳定定位的块索引（可选）
    query_index: int | None = None
    match_index: int | None = None
