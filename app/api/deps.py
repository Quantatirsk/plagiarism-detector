from functools import lru_cache
from app.services.embedding import EmbeddingService
from app.services.storage import MilvusStorage
from app.services.text_processor import TextProcessor
from app.services.document import DocumentService
from app.services.health import HealthService
from app.repositories.redis import RedisCache
from fastapi import Depends

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务单例"""
    return EmbeddingService()

@lru_cache()
def get_storage_service() -> MilvusStorage:
    """获取存储服务单例"""
    return MilvusStorage()

@lru_cache()
def get_text_processor() -> TextProcessor:
    """获取文本处理器单例"""
    return TextProcessor()

@lru_cache()
def get_cache_service() -> RedisCache:
    """获取缓存服务单例"""
    return RedisCache()


def get_document_service(
    embedding: EmbeddingService = Depends(get_embedding_service),
    storage: MilvusStorage = Depends(get_storage_service),
    processor: TextProcessor = Depends(get_text_processor)
) -> DocumentService:
    """组装文档服务"""
    return DocumentService(embedding, storage, processor)

def get_health_service(
    storage: MilvusStorage = Depends(get_storage_service),
    cache: RedisCache = Depends(get_cache_service)
) -> HealthService:
    """组装健康检查服务"""
    return HealthService(storage, cache)