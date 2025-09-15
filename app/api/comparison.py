"""
文档对比API路由

支持文件上传和双文档对比功能
"""
import os
import tempfile
from typing import Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.services.dual_document_detection import get_dual_detection_service
from app.services.document_parser import get_document_parser
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/comparison", tags=["Document Comparison"])

# 允许的文件扩展名与大小限制
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB


def _sanitize_filename(filename: str) -> str:
    """去除路径并限制扩展名"""
    name = Path(filename or "").name  # 移除目录穿越
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext or 'unknown'}")
    return name


async def _save_upload_to_temp(upload: UploadFile, prefix: str) -> str:
    """将上传文件安全保存到临时目录并返回路径，带大小限制"""
    safe_name = _sanitize_filename(upload.filename)
    ext = Path(safe_name).suffix
    # 使用安全的临时文件名，避免使用原始文件名
    tmp = tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=ext, delete=False, dir=tempfile.gettempdir())
    tmp_path = tmp.name
    bytes_written = 0
    try:
        CHUNK = 1024 * 1024
        while True:
            chunk = await upload.read(CHUNK)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File too large (max 10MB)")
            tmp.write(chunk)
        tmp.flush()
        return tmp_path
    except Exception:
        try:
            tmp.close()
        finally:
            # 出错时清理已写入的临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        raise
    finally:
        tmp.close()


@router.post("/upload-and-compare")
async def upload_and_compare_documents(
    document1: UploadFile = File(..., description="第一个文档文件"),
    document2: UploadFile = File(..., description="第二个文档文件"),
    granularity: str = Form("paragraph", description="检测粒度: paragraph/sentence"),
    threshold: float | None = Form(None, description="相似度阈值，可选"),
    top_k_per_query: int | None = Form(None, description="每个查询块最多返回的匹配数量，仅段落模式生效，可选"),
    max_total_matches: int | None = Form(None, description="最大匹配总数上限，可选")
) -> Dict[str, Any]:
    """
    上传并对比两个文档

    Args:
        document1: 第一个文档文件
        document2: 第二个文档文件
        granularity: 检测粒度（paragraph 或 sentence）
        threshold: 相似度阈值（可选，0..1；未提供按粒度默认）

    Returns:
        对比结果
    """
    temp_files = []

    try:
        # 验证参数
        if granularity not in ["paragraph", "sentence"]:
            raise HTTPException(status_code=400, detail="Invalid granularity, must be 'paragraph' or 'sentence'")
        if threshold is not None and not (0.0 <= threshold <= 1.0):
            raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        if top_k_per_query is not None and top_k_per_query <= 0:
            raise HTTPException(status_code=400, detail="top_k_per_query must be positive")
        if max_total_matches is not None and max_total_matches <= 0:
            raise HTTPException(status_code=400, detail="max_total_matches must be positive")

        # 保存上传文件（包含格式与大小校验）
        doc1_path = await _save_upload_to_temp(document1, "doc1")
        temp_files.append(doc1_path)
        doc2_path = await _save_upload_to_temp(document2, "doc2")
        temp_files.append(doc2_path)

        # 执行对比
        logger.info(f"Starting comparison with granularity={granularity}, threshold={threshold}")
        detection_service = get_dual_detection_service()
        result = await detection_service.compare_documents(
            doc1_path=doc1_path,
            doc2_path=doc2_path,
            granularity=granularity,
            threshold=threshold,
            top_k_per_query=top_k_per_query,
            max_total_matches=max_total_matches
        )

        logger.info(f"Document comparison completed successfully for {document1.filename} vs {document2.filename}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    finally:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


@router.post("/check-format")
async def check_document_format(
    document: UploadFile = File(..., description="要检查的文档文件")
) -> Dict[str, Any]:
    """
    检查文档格式是否支持

    Args:
        document: 文档文件

    Returns:
        格式检查结果
    """
    try:
        # 保存到临时文件
        temp_path = await _save_upload_to_temp(document, "check")
        try:
            doc_parser = get_document_parser()
            doc_info = doc_parser.get_document_info(temp_path)
            return {
                "filename": Path(document.filename or '').name,
                "is_supported": doc_info["is_supported"],
                "extension": doc_info["extension"],
                "size": doc_info["size"],
                "is_parseable": doc_info.get("is_parseable", False),
                "text_length": doc_info.get("text_length", 0)
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Format check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Format check failed: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    获取支持的文档格式列表

    Returns:
        支持的格式列表
    """
    try:
        # 统一以服务端上传白名单为准，避免前后端认知不一致
        allowed = sorted(ALLOWED_EXTENSIONS)
        return {
            "allowed_formats": allowed,
            "document_formats": sorted(set([ext for ext in allowed if ext in {'.pdf', '.docx', '.doc'}])),
            "text_formats": sorted(set([ext for ext in allowed if ext in {'.txt', '.md'}]))
        }

    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported formats")


def _is_supported_file(filename: str, doc_parser) -> bool:
    """检查文件是否支持（基于扩展名）"""
    if not filename:
        return False
    try:
        name = Path(filename).name
        ext = Path(name).suffix.lower()
        return ext in ALLOWED_EXTENSIONS
    except Exception:
        return False
