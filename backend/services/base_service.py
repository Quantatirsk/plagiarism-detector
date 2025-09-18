"""
基础服务类 - 所有服务的公共功能
Following Linus principle: Keep it simple and practical
"""
from typing import TypeVar, Type
from backend.core.logging import get_logger
from backend.core.config import get_settings

# Type variable for singleton decorator
T = TypeVar('T')


def singleton(cls: Type[T]) -> Type[T]:
    """
    单例装饰器 - 确保服务类只有一个实例
    Simple and practical implementation
    """
    instances = {}

    # Create a wrapper class that inherits from the original
    class SingletonWrapper(cls):  # type: ignore
        def __new__(cls, *args, **kwargs):
            if cls not in instances:
                instances[cls] = object.__new__(cls)
            return instances[cls]

        def __init__(self, *args, **kwargs):
            # Only initialize once
            if not hasattr(self, '_singleton_initialized'):
                super().__init__(*args, **kwargs)
                self._singleton_initialized = True

    # Preserve class metadata
    SingletonWrapper.__name__ = cls.__name__
    SingletonWrapper.__qualname__ = cls.__qualname__
    SingletonWrapper.__module__ = cls.__module__
    SingletonWrapper.__doc__ = cls.__doc__

    return SingletonWrapper  # type: ignore


class BaseService:
    """
    基础服务类 - 提供所有服务的公共功能

    Features:
    - Automatic logger initialization
    - Settings access
    - Common error handling patterns
    """

    def __init__(self):
        """初始化基础服务"""
        # Use the actual module name for the logger
        self.logger = get_logger(self.__class__.__module__)
        self.settings = get_settings()
        self._initialized = False

    def _ensure_initialized(self):
        """确保服务已初始化 - 子类可以覆盖此方法"""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        """子类可以覆盖此方法进行特定初始化"""
        pass

    def __repr__(self):
        """Simple representation for debugging"""
        return f"<{self.__class__.__name__} initialized={self._initialized}>"