"""
错误处理模块 - 定义自定义异常类和错误处理逻辑
遵循简单性原则：清晰的错误分类和有意义的错误消息
"""
from enum import Enum
from typing import Optional, Any, Dict
from fastapi import HTTPException, status


class ErrorCode(str, Enum):
    """错误代码枚举"""
    # 客户端错误
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_INPUT = "INVALID_INPUT"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"
    
    # 服务端错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # 外部服务错误
    OPENAI_ERROR = "OPENAI_ERROR"
    MILVUS_ERROR = "MILVUS_ERROR"
    REDIS_ERROR = "REDIS_ERROR"
    
    # 业务逻辑错误
    DETECTION_FAILED = "DETECTION_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    STORAGE_FAILED = "STORAGE_FAILED"
    TEXT_PROCESSING_FAILED = "TEXT_PROCESSING_FAILED"


class BaseApplicationError(Exception):
    """应用基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }


# 客户端错误 (4xx)
class InvalidRequestError(BaseApplicationError):
    """无效请求错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_REQUEST,
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class InvalidInputError(BaseApplicationError):
    """输入验证错误"""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_INPUT,
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class ResourceNotFoundError(BaseApplicationError):
    """资源不存在错误"""
    def __init__(self, resource_type: str, resource_id: Optional[str] = None):
        message = f"{resource_type} not found"
        details = {"resource_type": resource_type}
        if resource_id:
            details["resource_id"] = resource_id
            message = f"{resource_type} with id '{resource_id}' not found"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            details=details,
            status_code=status.HTTP_404_NOT_FOUND
        )


class DuplicateResourceError(BaseApplicationError):
    """资源重复错误"""
    def __init__(self, resource_type: str, identifier: str):
        super().__init__(
            message=f"{resource_type} with identifier '{identifier}' already exists",
            error_code=ErrorCode.DUPLICATE_RESOURCE,
            details={"resource_type": resource_type, "identifier": identifier},
            status_code=status.HTTP_409_CONFLICT
        )


# 服务端错误 (5xx)
class InternalServerError(BaseApplicationError):
    """内部服务器错误"""
    def __init__(self, message: str = "An internal error occurred", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class ServiceUnavailableError(BaseApplicationError):
    """服务不可用错误"""
    def __init__(self, service_name: str, reason: Optional[str] = None):
        message = f"{service_name} service is currently unavailable"
        details = {"service": service_name}
        if reason:
            message = f"{message}: {reason}"
            details["reason"] = reason
        
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            details=details,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


# 外部服务错误
class OpenAIError(BaseApplicationError):
    """OpenAI API错误"""
    def __init__(self, message: str, api_error: Optional[Exception] = None):
        details = {}
        if api_error:
            details["original_error"] = str(api_error)
        
        super().__init__(
            message=f"OpenAI API error: {message}",
            error_code=ErrorCode.OPENAI_ERROR,
            details=details,
            status_code=status.HTTP_502_BAD_GATEWAY
        )


class MilvusError(BaseApplicationError):
    """Milvus数据库错误"""
    def __init__(self, message: str, operation: Optional[str] = None, original_error: Optional[Exception] = None):
        details = {}
        if operation:
            details["operation"] = operation
        if original_error:
            details["original_error"] = str(original_error)
        
        super().__init__(
            message=f"Milvus error: {message}",
            error_code=ErrorCode.MILVUS_ERROR,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class RedisError(BaseApplicationError):
    """Redis缓存错误"""
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=f"Redis error: {message}",
            error_code=ErrorCode.REDIS_ERROR,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# 业务逻辑错误
class DetectionError(BaseApplicationError):
    """检测失败错误"""
    def __init__(self, message: str, stage: Optional[str] = None):
        details = {}
        if stage:
            details["stage"] = stage
        
        super().__init__(
            message=f"Detection failed: {message}",
            error_code=ErrorCode.DETECTION_FAILED,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class EmbeddingError(BaseApplicationError):
    """嵌入生成错误"""
    def __init__(self, message: str, text_length: Optional[int] = None):
        details = {}
        if text_length is not None:
            details["text_length"] = text_length
        
        super().__init__(
            message=f"Embedding generation failed: {message}",
            error_code=ErrorCode.EMBEDDING_FAILED,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class StorageError(BaseApplicationError):
    """存储操作错误"""
    def __init__(self, message: str, operation: str):
        super().__init__(
            message=f"Storage operation '{operation}' failed: {message}",
            error_code=ErrorCode.STORAGE_FAILED,
            details={"operation": operation},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class TextProcessingError(BaseApplicationError):
    """文本处理错误"""
    def __init__(self, message: str, process_type: Optional[str] = None):
        details = {}
        if process_type:
            details["process_type"] = process_type
        
        super().__init__(
            message=f"Text processing failed: {message}",
            error_code=ErrorCode.TEXT_PROCESSING_FAILED,
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


def create_http_exception(error: BaseApplicationError) -> HTTPException:
    """将自定义异常转换为FastAPI HTTPException"""
    return HTTPException(
        status_code=error.status_code,
        detail=error.to_dict()
    )


# 简单的别名 - 遵循Linus原则
APIError = BaseApplicationError