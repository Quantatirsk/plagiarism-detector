"""
服务模块 - 提供统一的服务访问接口
"""

# 导出基础服务类和装饰器
from backend.services.base_service import BaseService, singleton

# 导出服务工厂
from backend.services.service_factory import ServiceFactory

__all__ = [
    # 基础类
    'BaseService',
    'singleton',

    # 服务工厂
    'ServiceFactory',
]