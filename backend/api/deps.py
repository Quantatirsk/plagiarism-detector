from functools import lru_cache
from backend.services import ServiceFactory
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_storage import MilvusStorage
from backend.services.text_processor import TextProcessor
from backend.services.health_service import HealthService
from backend.repositories.redis import RedisCache
from fastapi import Depends

def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务单例"""
    return ServiceFactory.get_embedding_service()

def get_storage_service() -> MilvusStorage:
    """获取存储服务单例"""
    return ServiceFactory.get_storage_service()

def get_text_processor() -> TextProcessor:
    """获取文本处理器单例"""
    return ServiceFactory.get_text_processor()

@lru_cache()
def get_cache_service() -> RedisCache:
    """获取缓存服务单例"""
    return RedisCache()

def get_health_service(
    storage: MilvusStorage = Depends(get_storage_service),
    cache: RedisCache = Depends(get_cache_service)
) -> HealthService:
    """组装健康检查服务"""
    return HealthService(storage, cache)


def get_detection_service():
    """获取检测编排服务单例"""
    return ServiceFactory.get_detection_orchestrator()
