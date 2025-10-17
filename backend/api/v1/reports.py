"""
报告生成API端点 - 支持文档、对比和项目级报告生成
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from backend.models.report_models import (
    ReportType, ReportGenerationRequest, GeneratedReport, ReportProgress
)
from backend.services.report_generator_service import ReportGeneratorService
from backend.services.service_factory import ServiceFactory
from backend.core.errors import create_http_exception, LLMError
import structlog

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])
logger = structlog.get_logger()


class DocumentReportRequest(BaseModel):
    """文档报告请求"""
    document_id: str = Field(..., description="文档ID")
    language: str = Field(default="zh", description="报告语言")
    include_charts: bool = Field(default=True, description="是否包含图表")
    include_recommendations: bool = Field(default=True, description="是否包含建议")
    max_matches_detail: int = Field(default=20, description="最大匹配详情数量")
    llm_model: Optional[str] = Field(None, description="LLM模型")
    stream_response: bool = Field(default=False, description="是否流式响应")


class ComparisonReportRequest(BaseModel):
    """对比报告请求"""
    document_a_id: str = Field(..., description="文档A ID")
    document_b_id: str = Field(..., description="文档B ID")
    language: str = Field(default="zh", description="报告语言")
    include_charts: bool = Field(default=True, description="是否包含图表")
    include_recommendations: bool = Field(default=True, description="是否包含建议")
    llm_model: Optional[str] = Field(None, description="LLM模型")
    stream_response: bool = Field(default=False, description="是否流式响应")


class ProjectReportRequest(BaseModel):
    """项目报告请求"""
    project_id: str = Field(..., description="项目ID")
    language: str = Field(default="zh", description="报告语言")
    include_charts: bool = Field(default=True, description="是否包含图表")
    include_recommendations: bool = Field(default=True, description="是否包含建议")
    include_network_graph: bool = Field(default=True, description="是否包含网络图")
    llm_model: Optional[str] = Field(None, description="LLM模型")
    stream_response: bool = Field(default=False, description="是否流式响应")


def get_report_generator() -> ReportGeneratorService:
    """获取报告生成服务"""
    return ServiceFactory.get_report_generator()


@router.post("/document", summary="生成文档抄袭检测报告")
async def generate_document_report(
    request: DocumentReportRequest,
    report_generator: ReportGeneratorService = Depends(get_report_generator)
):
    """
    生成单个文档的抄袭检测报告

    分析指定文档在整个数据库中的抄袭情况，包括：
    - 总体相似度评估
    - 相似度来源分析
    - 高风险匹配详情
    - 风险等级和改进建议
    """
    try:
        # 构建报告生成请求
        generation_request = ReportGenerationRequest(
            type=ReportType.DOCUMENT,
            document_id=request.document_id,
            language=request.language,
            include_charts=request.include_charts,
            include_recommendations=request.include_recommendations,
            max_matches_detail=request.max_matches_detail,
            llm_model=request.llm_model,
            stream_response=request.stream_response
        )

        if request.stream_response:
            # 流式响应
            async def generate():
                try:
                    generator = await report_generator.generate_report(generation_request, stream=True)
                    async for chunk in generator:
                        yield f"data: {chunk}\n"
                except Exception as e:
                    error_data = {"type": "error", "message": str(e)}
                    yield f"data: {json.dumps(error_data)}\n"
                yield "data: [DONE]\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 标准响应
            report = await report_generator.generate_report(generation_request, stream=False)
            return report

    except ValueError as e:
        logger.error("Invalid request parameters", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        logger.error("LLM error in report generation", error=str(e))
        raise create_http_exception(e)
    except Exception as e:
        logger.error("Unexpected error in document report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/comparison", summary="生成文档对比分析报告")
async def generate_comparison_report(
    request: ComparisonReportRequest,
    report_generator: ReportGeneratorService = Depends(get_report_generator)
):
    """
    生成两个文档间的详细对比分析报告

    深入分析两个特定文档之间的相似性和差异，包括：
    - 双向相似度分析
    - 并排对比视图
    - 内容分布分析
    - 相似性模式识别
    """
    try:
        generation_request = ReportGenerationRequest(
            type=ReportType.COMPARISON,
            document_a_id=request.document_a_id,
            document_b_id=request.document_b_id,
            language=request.language,
            include_charts=request.include_charts,
            include_recommendations=request.include_recommendations,
            llm_model=request.llm_model,
            stream_response=request.stream_response
        )

        if request.stream_response:
            async def generate():
                try:
                    generator = await report_generator.generate_report(generation_request, stream=True)
                    async for chunk in generator:
                        yield f"data: {chunk}\n"
                except Exception as e:
                    error_data = {"type": "error", "message": str(e)}
                    yield f"data: {json.dumps(error_data)}\n"
                yield "data: [DONE]\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            report = await report_generator.generate_report(generation_request, stream=False)
            return report

    except ValueError as e:
        logger.error("Invalid request parameters", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        logger.error("LLM error in comparison report generation", error=str(e))
        raise create_http_exception(e)
    except Exception as e:
        logger.error("Unexpected error in comparison report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.post("/project", summary="生成项目学术诚信分析报告")
async def generate_project_report(
    request: ProjectReportRequest,
    report_generator: ReportGeneratorService = Depends(get_report_generator)
):
    """
    生成项目级综合学术诚信分析报告

    对整个项目（多个文档的集合）进行宏观分析，包括：
    - 项目整体统计概览
    - 相似度分布和趋势分析
    - 高风险文档识别
    - 群体行为模式分析
    - 异常检测和预警
    """
    try:
        generation_request = ReportGenerationRequest(
            type=ReportType.PROJECT,
            project_id=request.project_id,
            language=request.language,
            include_charts=request.include_charts,
            include_recommendations=request.include_recommendations,
            include_network_graph=request.include_network_graph,
            llm_model=request.llm_model,
            stream_response=request.stream_response
        )

        if request.stream_response:
            async def generate():
                try:
                    generator = await report_generator.generate_report(generation_request, stream=True)
                    async for chunk in generator:
                        yield f"data: {chunk}\n"
                except Exception as e:
                    error_data = {"type": "error", "message": str(e)}
                    yield f"data: {json.dumps(error_data)}\n"
                yield "data: [DONE]\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            report = await report_generator.generate_report(generation_request, stream=False)
            return report

    except ValueError as e:
        logger.error("Invalid request parameters", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        logger.error("LLM error in project report generation", error=str(e))
        raise create_http_exception(e)
    except Exception as e:
        logger.error("Unexpected error in project report generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/progress/{task_id}", summary="获取报告生成进度")
async def get_report_progress(
    task_id: str,
    report_generator: ReportGeneratorService = Depends(get_report_generator)
):
    """
    获取报告生成的实时进度

    返回指定任务的生成进度，包括：
    - 完成百分比
    - 当前处理阶段
    - 预计剩余时间
    - 错误信息（如有）
    """
    try:
        progress = report_generator.get_generation_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found or completed")

        return progress

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving report progress", error=str(e), task_id=task_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve progress")


@router.delete("/progress/{task_id}", summary="取消报告生成")
async def cancel_report_generation(
    task_id: str,
    report_generator: ReportGeneratorService = Depends(get_report_generator)
):
    """
    取消正在进行的报告生成任务

    立即停止指定的报告生成任务，释放相关资源
    """
    try:
        success = await report_generator.cancel_generation(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or already completed")

        return {"message": "Report generation cancelled successfully", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error cancelling report generation", error=str(e), task_id=task_id)
        raise HTTPException(status_code=500, detail="Failed to cancel generation")


@router.get("/templates", summary="获取可用报告模板")
async def get_available_templates():
    """
    获取系统中可用的报告模板列表

    返回所有支持的报告类型和语言组合
    """
    try:
        from backend.services.report_template_service import ReportTemplateService
        template_service = ReportTemplateService()
        templates = template_service.get_available_templates()

        return {
            "templates": templates,
            "supported_languages": ["zh", "en"],
            "supported_types": [t.value for t in ReportType]
        }

    except Exception as e:
        logger.error("Error retrieving templates", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@router.get("/health", summary="报告服务健康检查")
async def report_service_health():
    """
    检查报告生成服务的健康状态

    验证所有依赖服务（LLM、模板、数据处理）的可用性
    """
    try:
        report_generator = ServiceFactory.get_report_generator()

        # 检查服务是否初始化
        if not hasattr(report_generator, '_initialized') or not report_generator._initialized:
            return {
                "status": "initializing",
                "message": "Report generator is initializing"
            }

        # TODO: 添加更详细的健康检查
        # - LLM服务连通性
        # - 模板服务状态
        # - 数据处理服务状态

        return {
            "status": "healthy",
            "message": "Report generation service is operational",
            "active_generations": len(report_generator.active_generations),
            "timestamp": json.loads(json.dumps({"time": "now"}, default=str))
        }

    except Exception as e:
        logger.error("Report service health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "message": f"Service health check failed: {str(e)}"
        }