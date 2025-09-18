from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.api.v1 import health, documents, compare, projects
from backend.core.config import get_settings
from backend.core.middleware import error_handler
from backend.db import init_db
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

settings = get_settings()
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(
        "Starting application",
        version=settings.version,
        mode=settings.milvus_mode,
        api_prefix=settings.api_v1_prefix
    )
    
    # 初始化资源
    try:
        await init_db()
        logger.info("Application startup complete")
        yield
    finally:
        # 关闭时
        logger.info("Shutting down application")
        # 这里可以添加清理逻辑
        # 例如：关闭连接池、保存状态等

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 中间件配置 - 安全的 CORS 设置
origins = settings.get_cors_origins()
# 浏览器规范：当 allow_origins 为 "*" 时，不能允许 credentials
allow_credentials = settings.cors_allow_credentials and origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 错误处理
app.add_exception_handler(Exception, error_handler)

# 路由注册
app.include_router(
    health.router,
    prefix=f"{settings.api_v1_prefix}/health",
    tags=["health"]
)

# API routers
app.include_router(documents.router)
app.include_router(compare.router)
app.include_router(projects.router)

# Prometheus监控
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.project_name,
        "version": settings.version,
        "docs": "/docs",
        "api_prefix": settings.api_v1_prefix,
        "mode": settings.milvus_mode.value
    }

@app.get(f"{settings.api_v1_prefix}")
async def api_root():
    """API根路径"""
    return {
        "version": "v1",
        "endpoints": {
            "health": f"{settings.api_v1_prefix}/health",
            "projects": "/api/v1/projects",
            "documents": "/api/v1/documents",
            "compare_jobs": "/api/v1/compare-jobs",
        }
    }


if __name__ == "__main__":
    # 简化版本：直接调用 run.py 来处理端口清理等复杂逻辑
    import subprocess
    import sys
    from pathlib import Path
    
    # 获取 run.py 的路径
    run_script = Path(__file__).parent.parent / "run.py"
    
    if run_script.exists():
        # 使用 run.py 启动，它会处理端口清理等
        print("Starting application via run.py...")
        subprocess.run([sys.executable, str(run_script)])
    else:
        # 简单的备用启动
        import uvicorn
        import os
        
        # 添加项目根目录到Python路径
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        port = int(os.getenv("PORT", 8000))
        
        print(f"🚀 Starting FastAPI application on http://0.0.0.0:{port}")
        print(f"📚 API documentation: http://localhost:{port}/docs")
        print("Press CTRL+C to stop\n")
        
        try:
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=port,
                reload=True,
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
            sys.exit(0)
