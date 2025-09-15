"""
健康检查服务 - 应用健康状态和就绪状态检查
"""
from typing import Dict, Any
from datetime import datetime
from app.core.config import get_settings, MilvusMode
from app.core.logging import get_logger
import openai

logger = get_logger(__name__)
settings = get_settings()


class HealthService:
    """健康检查服务 - 监控系统状态"""
    
    def __init__(self, storage=None, cache=None):
        self.start_time = datetime.now()
        self.storage = storage
        self.cache = cache
    
    async def check_health(self) -> Dict[str, Any]:
        """基础健康检查 - 应用是否运行"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "version": settings.version,
            "mode": "development" if settings.milvus_mode == MilvusMode.LOCAL else "production"
        }
    
    async def check_readiness(self) -> Dict[str, Any]:
        """就绪检查 - 系统是否准备好接收请求"""
        checks = {
            "api": True,  # API总是就绪的
            "milvus": False,
            "openai": False,
            "overall": False
        }
        
        errors = []
        
        # 检查Milvus连接
        try:
            checks["milvus"] = await self._check_milvus()
        except Exception as e:
            logger.error("Milvus check failed", error=str(e))
            errors.append(f"Milvus: {str(e)}")
        
        # 检查OpenAI API
        try:
            checks["openai"] = await self._check_openai()
        except Exception as e:
            logger.error("OpenAI check failed", error=str(e))
            errors.append(f"OpenAI: {str(e)}")
        
        # 总体就绪状态
        checks["overall"] = checks["milvus"] and checks["openai"]
        
        result = {
            "ready": checks["overall"],
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
        
        if errors:
            result["errors"] = errors
        
        return result
    
    async def _check_milvus(self) -> bool:
        """检查Milvus连接状态"""
        try:
            if settings.milvus_mode == MilvusMode.LOCAL:
                # 本地模式 - 检查文件是否可访问
                from pymilvus import MilvusClient
                client = MilvusClient(settings.milvus_db_file)
                # 尝试列出集合
                collections = client.list_collections()
                return True
            else:
                # 服务器模式 - 检查连接
                from pymilvus import connections
                connections.connect(
                    alias="health_check",
                    host=settings.milvus_host,
                    port=settings.milvus_port
                )
                connections.disconnect("health_check")
                return True
        except Exception as e:
            logger.error("Milvus health check failed", error=str(e))
            return False
    
    async def _check_openai(self) -> bool:
        """检查OpenAI API连接"""
        try:
            client = openai.AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
            
            # 尝试获取模型列表或进行简单的嵌入测试
            # 使用一个很短的测试文本以减少成本
            response = await client.embeddings.create(
                input="test",
                model=settings.openai_model
            )
            
            return len(response.data) > 0
        except Exception as e:
            logger.error("OpenAI health check failed", error=str(e))
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """获取系统详细信息"""
        return {
            "configuration": {
                "milvus_mode": settings.milvus_mode,
                "openai_model": settings.openai_model,
                "openai_dimensions": settings.openai_dimensions,
                "batch_size": settings.openai_batch_size,
                "collection_name": settings.milvus_collection,
                "paragraph_threshold": settings.paragraph_similarity_threshold,
                "sentence_threshold": settings.sentence_similarity_threshold,
                "top_k_paragraphs": settings.top_k_paragraphs
            },
            "limits": {
                "max_concurrent_requests": settings.max_concurrent_requests,
                "request_timeout": settings.request_timeout
            },
            "api": {
                "version": settings.version,
                "prefix": settings.api_v1_prefix
            }
        }