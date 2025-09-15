from fastapi import APIRouter, Depends, HTTPException
from app.services.health import HealthService
from app.api.deps import get_health_service
from typing import Dict, Any
import structlog

router = APIRouter()
logger = structlog.get_logger()

@router.get("/")
async def health_check(
    health_service: HealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    健康检查
    
    返回应用的基本健康状态
    """
    try:
        health_status = await health_service.check_health()
        return health_status
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")

@router.get("/ready")
async def readiness_check(
    health_service: HealthService = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    就绪检查
    
    检查所有依赖服务是否就绪
    """
    try:
        readiness_status = await health_service.check_readiness()
        
        # 如果任何服务未就绪，返回503
        if not readiness_status.get("ready", False):
            raise HTTPException(
                status_code=503, 
                detail=f"Service not ready: {readiness_status.get('reason', 'Unknown')}"
            )
        
        return readiness_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    存活检查
    
    简单的存活探针，用于Kubernetes等容器编排工具
    """
    return {"status": "alive"}