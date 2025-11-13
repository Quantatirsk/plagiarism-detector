"""
报告数据模型 - 支持文档、对比和项目级报告
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReportType(str, Enum):
    """报告类型枚举"""
    DOCUMENT = "document"      # 单文档分析
    COMPARISON = "comparison"  # 双文档对比
    PROJECT = "project"       # 项目级分析


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"           # 低风险 (0-20%)
    MODERATE = "moderate" # 中等风险 (20-50%)
    HIGH = "high"        # 高风险 (50-80%)
    CRITICAL = "critical" # 严重风险 (80%+)


class SimilaritySource(BaseModel):
    """相似度来源信息"""
    document_id: str = Field(..., description="来源文档ID")
    document_title: str = Field(..., description="来源文档标题")
    similarity_score: float = Field(..., description="相似度分数 (0-1)")
    match_count: int = Field(..., description="匹配数量")
    total_text_length: int = Field(..., description="匹配文本总长度")


class MatchDetail(BaseModel):
    """匹配详情"""
    source_text: str = Field(..., description="源文本")
    target_text: str = Field(..., description="目标文本")
    similarity_score: float = Field(..., description="相似度")
    source_start: int = Field(..., description="源文本起始位置")
    source_end: int = Field(..., description="源文本结束位置")
    target_start: int = Field(..., description="目标文本起始位置")
    target_end: int = Field(..., description="目标文本结束位置")
    match_type: str = Field(default="semantic", description="匹配类型")


class DocumentReportData(BaseModel):
    """文档报告数据"""
    document_id: str = Field(..., description="文档ID")
    document_title: str = Field(..., description="文档标题")
    total_similarity_score: float = Field(..., description="总体相似度")
    risk_level: RiskLevel = Field(..., description="风险等级")
    sources: list[SimilaritySource] = Field(default_factory=list, description="相似度来源列表")
    top_matches: list[MatchDetail] = Field(default_factory=list, description="高相似度匹配")
    statistics: dict[str, Any] = Field(default_factory=dict, description="统计信息")


class ComparisonReportData(BaseModel):
    """对比报告数据"""
    document_a_id: str = Field(..., description="文档A ID")
    document_b_id: str = Field(..., description="文档B ID")
    document_a_title: str = Field(..., description="文档A标题")
    document_b_title: str = Field(..., description="文档B标题")
    similarity_a_to_b: float = Field(..., description="A→B相似度")
    similarity_b_to_a: float = Field(..., description="B→A相似度")
    common_similarity: float = Field(..., description="共同相似度")
    unique_a_ratio: float = Field(..., description="A独有内容比例")
    unique_b_ratio: float = Field(..., description="B独有内容比例")
    match_details: list[MatchDetail] = Field(default_factory=list, description="匹配详情")
    side_by_side_sections: list[dict[str, Any]] = Field(default_factory=list, description="并排对比数据")


class ProjectStatistics(BaseModel):
    """项目统计信息"""
    total_documents: int = Field(..., description="文档总数")
    total_comparisons: int = Field(..., description="比较总数")
    average_similarity: float = Field(..., description="平均相似度")
    high_risk_count: int = Field(..., description="高风险文档数")
    similarity_distribution: dict[str, int] = Field(default_factory=dict, description="相似度分布")
    most_similar_pairs: list[dict[str, Any]] = Field(default_factory=list, description="最相似文档对")


class ProjectReportData(BaseModel):
    """项目报告数据"""
    project_id: str = Field(..., description="项目ID")
    project_name: str = Field(..., description="项目名称")
    statistics: ProjectStatistics = Field(..., description="项目统计")
    high_risk_documents: list[DocumentReportData] = Field(default_factory=list, description="高风险文档")
    similarity_network: dict[str, Any] = Field(default_factory=dict, description="相似度网络图数据")
    anomalies: list[dict[str, Any]] = Field(default_factory=list, description="异常检测结果")
    recommendations: list[str] = Field(default_factory=list, description="建议列表")


class GeneratedReport(BaseModel):
    """生成的报告"""
    id: str = Field(..., description="报告ID")
    type: ReportType = Field(..., description="报告类型")
    title: str = Field(..., description="报告标题")
    summary: str = Field(..., description="报告摘要")
    content: dict[str, Any] = Field(..., description="报告内容")
    data: DocumentReportData | ComparisonReportData | ProjectReportData = Field(..., description="报告数据")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="生成时间")
    generated_by: str = Field(default="system", description="生成者")
    language: str = Field(default="zh", description="报告语言")
    export_formats: list[str] = Field(default_factory=lambda: ["html", "pdf", "json"], description="支持的导出格式")

    class Config:
        json_encoders = {
            datetime: lambda value: value.isoformat()
        }


class ReportGenerationRequest(BaseModel):
    """报告生成请求"""
    type: ReportType = Field(..., description="报告类型")
    language: str = Field(default="zh", description="报告语言 (zh/en)")
    include_charts: bool = Field(default=True, description="是否包含图表")
    include_recommendations: bool = Field(default=True, description="是否包含建议")
    max_matches_detail: int = Field(default=20, description="最大匹配详情数量")

    # 文档报告参数
    document_id: str | None = Field(default=None, description="文档ID (文档报告)")

    # 对比报告参数
    document_a_id: str | None = Field(default=None, description="文档A ID (对比报告)")
    document_b_id: str | None = Field(default=None, description="文档B ID (对比报告)")

    # 项目报告参数
    project_id: str | None = Field(default=None, description="项目ID (项目报告)")
    include_network_graph: bool = Field(default=True, description="是否包含网络图")

    # LLM生成参数
    llm_model: str | None = Field(None, description="LLM模型")
    stream_response: bool = Field(default=False, description="是否流式响应")


class ReportTemplate(BaseModel):
    """报告模板"""
    type: ReportType = Field(..., description="报告类型")
    language: str = Field(..., description="模板语言")
    system_prompt: str = Field(..., description="系统提示词")
    user_prompt_template: str = Field(..., description="用户提示词模板")
    sections: list[str] = Field(..., description="报告章节列表")
    chart_configs: dict[str, Any] = Field(default_factory=dict, description="图表配置")


class ReportProgress(BaseModel):
    """报告生成进度"""
    task_id: str = Field(..., description="任务ID")
    progress: float = Field(..., description="进度百分比 (0-1)")
    stage: str = Field(..., description="当前阶段")
    message: str = Field(..., description="进度信息")
    estimated_remaining: int | None = Field(default=None, description="预计剩余时间(秒)")
    error: str | None = Field(default=None, description="错误信息")
