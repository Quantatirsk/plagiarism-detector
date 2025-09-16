"""
健康检查服务 - 应用健康状态和就绪状态检查
"""
from typing import Dict, Any
from datetime import datetime
from app.core.config import MilvusMode
from app.services.base_service import BaseService
import openai


class HealthService(BaseService):
    """健康检查服务 - 监控系统状态"""

    def __init__(self, storage=None, cache=None):
        super().__init__()
        self.storage = storage
        self.cache = cache

    def _initialize(self):
        """初始化健康检查服务"""
        self.start_time = datetime.now()
    
    async def check_health(self) -> Dict[str, Any]:
        """基础健康检查 - 应用是否运行"""
        self._ensure_initialized()
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "version": self.settings.version,
            "mode": "development" if self.settings.milvus_mode == MilvusMode.LOCAL else "production"
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
            self.logger.error("Milvus check failed", error=str(e))
            errors.append(f"Milvus: {str(e)}")
        
        # 检查OpenAI API
        try:
            checks["openai"] = await self._check_openai()
        except Exception as e:
            self.logger.error("OpenAI check failed", error=str(e))
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
            if self.settings.milvus_mode == MilvusMode.LOCAL:
                # 本地模式 - 检查文件是否可访问
                from pymilvus import MilvusClient
                client = MilvusClient(self.settings.milvus_db_file)
                # 尝试列出集合
                client.list_collections()  # 只是检查是否能连接
                return True
            else:
                # 服务器模式 - 检查连接
                from pymilvus import connections
                connections.connect(
                    alias="health_check",
                    host=self.settings.milvus_host,
                    port=self.settings.milvus_port
                )
                connections.disconnect("health_check")
                return True
        except Exception as e:
            self.logger.error("Milvus health check failed", error=str(e))
            return False
    
    async def _check_openai(self) -> bool:
        """检查OpenAI API连接"""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url
            )

            # 尝试获取模型列表或进行简单的嵌入测试
            # 使用一个很短的测试文本以减少成本
            response = await client.embeddings.create(
                input="test",
                model=self.settings.openai_model
            )
            
            return len(response.data) > 0
        except Exception as e:
            self.logger.error("OpenAI health check failed", error=str(e))
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """获取系统详细信息"""
        return {
            "configuration": {
                "milvus_mode": self.settings.milvus_mode,
                "openai_model": self.settings.openai_model,
                "openai_dimensions": self.settings.openai_dimensions,
                "batch_size": self.settings.openai_batch_size,
                "collection_name": self.settings.milvus_collection,
                "paragraph_threshold": self.settings.paragraph_similarity_threshold,
                "sentence_threshold": self.settings.sentence_similarity_threshold,
                "top_k_paragraphs": self.settings.top_k_paragraphs
            },
            "limits": {
                "max_concurrent_requests": self.settings.max_concurrent_requests,
                "request_timeout": self.settings.request_timeout
            },
            "api": {
                "version": self.settings.version,
                "prefix": self.settings.api_v1_prefix
            }
        }