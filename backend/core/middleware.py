"""
中间件模块 - 全局错误处理和请求/响应处理
遵循清晰性原则：统一的错误响应格式
"""
import time
import traceback
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.errors import BaseApplicationError, InternalServerError, create_http_exception


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局错误处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并捕获所有异常"""
        # 生成请求ID用于追踪
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            
            # 记录处理时间
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except BaseApplicationError as e:
            # 处理自定义应用异常
            return self._create_error_response(
                request_id=request_id,
                status_code=e.status_code,
                error_code=e.error_code.value,
                message=e.message,
                details=e.details
            )
            
        except ValueError as e:
            # 处理值错误（通常是输入验证问题）
            return self._create_error_response(
                request_id=request_id,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="INVALID_INPUT",
                message=str(e),
                details={"type": "ValueError"}
            )
            
        except Exception as e:
            # 处理未预期的异常
            # 在开发模式下，返回详细的错误信息
            details = {
                "type": type(e).__name__,
                "traceback": traceback.format_exc() if self._is_development() else None
            }
            
            # 记录错误（这里简化处理，实际应该使用日志系统）
            print(f"[ERROR] Request {request_id}: {e}")
            if self._is_development():
                print(traceback.format_exc())
            
            return self._create_error_response(
                request_id=request_id,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                details=details
            )
    
    def _create_error_response(
        self,
        request_id: str,
        status_code: int,
        error_code: str,
        message: str,
        details: dict = None
    ) -> JSONResponse:
        """创建统一的错误响应"""
        content = {
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "request_id": request_id
            }
        }
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers={"X-Request-ID": request_id}
        )
    
    def _is_development(self) -> bool:
        """判断是否为开发环境"""
        # 简化处理，实际应该从配置中读取
        import os
        return os.getenv("ENVIRONMENT", "development") == "development"


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """请求验证中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """验证请求的基本要素"""
        # 检查Content-Type
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # API端点应该接受JSON
            if request.url.path.startswith("/api/") and not content_type.startswith("application/json"):
                # 除非是文件上传
                if not content_type.startswith("multipart/form-data"):
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "error": {
                                "code": "UNSUPPORTED_MEDIA_TYPE",
                                "message": "Content-Type must be application/json or multipart/form-data"
                            }
                        }
                    )
        
        # 继续处理请求
        response = await call_next(request)
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS处理中间件（简化版）"""
    
    def __init__(self, app, allow_origins: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """添加CORS响应头"""
        # 处理OPTIONS请求
        if request.method == "OPTIONS":
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Max-Age": "3600"
                }
            )
        
        # 处理常规请求
        response = await call_next(request)
        
        # 添加CORS头
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response


# 简单的错误处理函数 - 遵循Linus原则
async def error_handler(request: Request, exc: Exception):
    """全局错误处理"""
    from fastapi import HTTPException
    
    if isinstance(exc, BaseApplicationError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message}
        )
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    else:
        # 未处理的异常
        import os
        error_detail = str(exc) if os.getenv("ENVIRONMENT", "development") == "development" else "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"error": error_detail}
        )