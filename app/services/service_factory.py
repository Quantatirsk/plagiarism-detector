"""
服务工厂 - 统一的服务创建和管理
Following Linus principle: Simple and practical service management
"""
from typing import TYPE_CHECKING

# 避免循环导入，使用TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService
    from app.services.vector_storage import MilvusStorage
    from app.services.text_processor import TextProcessor
    from app.services.document_parser import DocumentParser
    from app.services.health_service import HealthService
    from app.services.dual_document_detection import DualDocumentDetectionService


class ServiceFactory:
    """
    服务工厂 - 提供统一的服务访问接口

    所有服务都使用单例模式，通过各自的 get_xxx_service() 函数获取
    """

    @staticmethod
    def get_embedding_service() -> 'EmbeddingService':
        """获取嵌入服务"""
        from app.services.embedding_service import EmbeddingService
        return EmbeddingService()

    @staticmethod
    def get_storage_service() -> 'MilvusStorage':
        """获取存储服务"""
        from app.services.vector_storage import MilvusStorage
        return MilvusStorage()

    @staticmethod
    def get_text_processor() -> 'TextProcessor':
        """获取文本处理器"""
        from app.services.text_processor import TextProcessor
        return TextProcessor()

    @staticmethod
    def get_document_parser() -> 'DocumentParser':
        """获取文档解析器"""
        from app.services.document_parser import DocumentParser
        return DocumentParser()

    @staticmethod
    def get_health_service(storage=None, cache=None) -> 'HealthService':
        """获取健康检查服务"""
        from app.services.health_service import HealthService
        # 健康检查服务需要传入依赖，不是纯单例
        return HealthService(storage=storage, cache=cache)

    @staticmethod
    def get_dual_detection_service() -> 'DualDocumentDetectionService':
        """获取双文档检测服务"""
        from app.services.dual_document_detection import DualDocumentDetectionService
        return DualDocumentDetectionService()