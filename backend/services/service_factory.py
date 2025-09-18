"""
服务工厂 - 统一的服务创建和管理
Following Linus principle: Simple and practical service management
"""
from typing import TYPE_CHECKING

# 避免循环导入，使用TYPE_CHECKING
if TYPE_CHECKING:
    from backend.services.embedding_service import EmbeddingService
    from backend.services.vector_storage import MilvusStorage
    from backend.services.text_processor import TextProcessor
    from backend.services.document_parser import DocumentParser
    from backend.services.health_service import HealthService
    from backend.services.detection_orchestrator import DetectionOrchestrator
    from backend.services.comparison_service import ComparisonService


class ServiceFactory:
    """
    服务工厂 - 提供统一的服务访问接口

    所有服务都使用单例模式，通过各自的 get_xxx_service() 函数获取
    """

    @staticmethod
    def get_embedding_service() -> 'EmbeddingService':
        """获取嵌入服务"""
        from backend.services.embedding_service import EmbeddingService
        return EmbeddingService()

    @staticmethod
    def get_storage_service() -> 'MilvusStorage':
        """获取存储服务"""
        from backend.services.vector_storage import MilvusStorage
        return MilvusStorage()

    @staticmethod
    def get_text_processor() -> 'TextProcessor':
        """获取文本处理器"""
        from backend.services.text_processor import TextProcessor
        return TextProcessor()

    @staticmethod
    def get_document_parser() -> 'DocumentParser':
        """获取文档解析器"""
        from backend.services.document_parser import DocumentParser
        return DocumentParser()

    @staticmethod
    def get_health_service(storage=None, cache=None) -> 'HealthService':
        """获取健康检查服务"""
        from backend.services.health_service import HealthService
        # 健康检查服务需要传入依赖，不是纯单例
        return HealthService(storage=storage, cache=cache)

    @staticmethod
    def get_detection_orchestrator() -> 'DetectionOrchestrator':
        """获取多文档检测调度服务"""
        from backend.services.detection_orchestrator import DetectionOrchestrator
        return DetectionOrchestrator()

    @staticmethod
    def get_comparison_service() -> 'ComparisonService':
        """获取对比执行服务"""
        from backend.services.comparison_service import ComparisonService
        return ComparisonService()
