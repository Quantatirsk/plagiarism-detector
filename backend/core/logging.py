"""
结构化日志配置模块 - 使用structlog实现JSON格式日志
遵循清晰性原则：日志即文档，提供有意义的上下文
"""
import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import FilteringBoundLogger, Processor


def configure_logging(
    level: str = "INFO",
    json_logs: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    配置结构化日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: 是否输出JSON格式日志
        log_file: 日志文件路径（可选）
    """
    
    # 设置Python标准日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 配置处理器链
    processors: list[Processor] = [
        # 添加时间戳
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        
        # 添加调用位置信息
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        
        # 添加异常信息格式化
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        
        # 格式化异常
        structlog.processors.format_exc_info,
    ]
    
    # 根据环境选择渲染器
    if json_logs:
        # 生产环境：JSON格式
        processors.append(structlog.processors.JSONRenderer())
    else:
        # 开发环境：彩色控制台输出
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.better_traceback
            )
        )
    
    # 配置structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库日志
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # 文件始终使用JSON格式
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        file_handler.setFormatter(json_formatter)
        
        logging.getLogger().addHandler(file_handler)


def get_logger(
    name: str,
    **initial_context: Any
) -> FilteringBoundLogger:
    """
    获取结构化日志记录器
    
    Args:
        name: 日志记录器名称（通常使用模块名）
        **initial_context: 初始上下文数据
    
    Returns:
        配置好的日志记录器
    """
    logger = structlog.get_logger(name)
    
    # 绑定初始上下文
    if initial_context:
        logger = logger.bind(**initial_context)
    
    return logger


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: FilteringBoundLogger, **context: Any):
        self.logger = logger
        self.context = context
        self.original_logger = None
    
    def __enter__(self) -> FilteringBoundLogger:
        """进入上下文，添加额外的上下文信息"""
        self.original_logger = self.logger
        return self.logger.bind(**self.context)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原始日志记录器"""
        # 上下文会自动清理，无需特殊处理
        pass


def log_function_call(logger: FilteringBoundLogger):
    """
    函数调用日志装饰器
    
    Args:
        logger: 日志记录器
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 记录函数调用
            logger.info(
                "function_called",
                function=func.__name__,
                args=str(args)[:100],  # 限制长度避免日志过大
                kwargs=str(kwargs)[:100]
            )
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录成功
                logger.info(
                    "function_completed",
                    function=func.__name__,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 记录失败
                logger.error(
                    "function_failed",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_async_function_call(logger: FilteringBoundLogger):
    """
    异步函数调用日志装饰器
    
    Args:
        logger: 日志记录器
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 记录函数调用
            logger.info(
                "async_function_called",
                function=func.__name__,
                args=str(args)[:100],
                kwargs=str(kwargs)[:100]
            )
            
            try:
                # 执行异步函数
                result = await func(*args, **kwargs)
                
                # 记录成功
                logger.info(
                    "async_function_completed",
                    function=func.__name__,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 记录失败
                logger.error(
                    "async_function_failed",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# 预定义的日志事件类型
class LogEvent:
    """标准化的日志事件类型"""
    
    # 应用生命周期
    APP_STARTED = "app_started"
    APP_STOPPED = "app_stopped"
    APP_ERROR = "app_error"
    
    # API请求
    REQUEST_RECEIVED = "request_received"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"
    
    # 外部服务
    OPENAI_CALL = "openai_api_call"
    OPENAI_SUCCESS = "openai_api_success"
    OPENAI_ERROR = "openai_api_error"
    
    MILVUS_OPERATION = "milvus_operation"
    MILVUS_SUCCESS = "milvus_success"
    MILVUS_ERROR = "milvus_error"
    
    REDIS_OPERATION = "redis_operation"
    REDIS_SUCCESS = "redis_success"
    REDIS_ERROR = "redis_error"
    
    # 业务逻辑
    DETECTION_STARTED = "detection_started"
    DETECTION_COMPLETED = "detection_completed"
    DETECTION_FAILED = "detection_failed"
    
    EMBEDDING_STARTED = "embedding_started"
    EMBEDDING_COMPLETED = "embedding_completed"
    EMBEDDING_FAILED = "embedding_failed"
    
    TEXT_PROCESSING_STARTED = "text_processing_started"
    TEXT_PROCESSING_COMPLETED = "text_processing_completed"
    TEXT_PROCESSING_FAILED = "text_processing_failed"
    
    # 性能指标
    PERFORMANCE_METRIC = "performance_metric"
    SLOW_OPERATION = "slow_operation"
    
    # 安全事件
    SECURITY_WARNING = "security_warning"
    AUTHENTICATION_FAILED = "authentication_failed"


def create_request_logger(request_id: str) -> FilteringBoundLogger:
    """
    创建请求级别的日志记录器
    
    Args:
        request_id: 请求ID
    
    Returns:
        绑定了请求ID的日志记录器
    """
    return get_logger("request", request_id=request_id)


def create_service_logger(service_name: str) -> FilteringBoundLogger:
    """
    创建服务级别的日志记录器
    
    Args:
        service_name: 服务名称
    
    Returns:
        绑定了服务名称的日志记录器
    """
    return get_logger(f"service.{service_name}", service=service_name)


# 默认日志记录器
default_logger = get_logger("app")