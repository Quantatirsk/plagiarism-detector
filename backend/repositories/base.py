"""
基础仓库模式 - 定义通用的数据访问接口和基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from datetime import datetime, timedelta

from backend.core.logging import get_logger

logger = get_logger(__name__)

# 泛型类型
T = TypeVar('T')
K = TypeVar('K')  # Key type


class BaseRepository(ABC, Generic[T, K]):
    """基础仓库抽象类 - 定义通用CRUD操作接口"""
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """创建实体"""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: K) -> Optional[T]:
        """根据ID获取实体"""
        pass
    
    @abstractmethod
    async def update(self, entity_id: K, updates: Dict[str, Any]) -> Optional[T]:
        """更新实体"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: K) -> bool:
        """删除实体"""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """列出所有实体"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """统计实体数量"""
        pass


class CacheableRepository(BaseRepository[T, K]):
    """支持缓存的仓库基类"""
    
    def __init__(self, cache_prefix: str = "", default_ttl: int = 3600):
        self.cache_prefix = cache_prefix
        self.default_ttl = default_ttl
    
    def _make_cache_key(self, key: Union[str, K]) -> str:
        """生成缓存键"""
        return f"{self.cache_prefix}:{key}" if self.cache_prefix else str(key)
    
    def _make_list_cache_key(self, **filters) -> str:
        """生成列表缓存键"""
        filter_str = ":".join(f"{k}={v}" for k, v in sorted(filters.items()))
        return f"{self.cache_prefix}:list:{filter_str}" if filter_str else f"{self.cache_prefix}:list:all"
    
    async def invalidate_cache(self, entity_id: K):
        """无效化相关缓存"""
        from backend.repositories.redis import get_redis
        
        try:
            redis = get_redis()
            cache_key = self._make_cache_key(entity_id)
            await redis.delete(cache_key)
            
            # 清除列表缓存
            list_pattern = f"{self.cache_prefix}:list:*"
            await redis.clear_pattern(list_pattern)
            
            logger.debug("缓存失效", entity_id=entity_id, cache_key=cache_key)
            
        except Exception as e:
            logger.warning("缓存失效失败", entity_id=entity_id, error=str(e))
    
    async def get_cached(self, key: str, default: Any = None) -> Any:
        """从缓存获取数据"""
        from backend.repositories.redis import get_redis
        
        try:
            redis = get_redis()
            return await redis.get(key, default=default)
        except Exception as e:
            logger.warning("缓存读取失败", key=key, error=str(e))
            return default
    
    async def set_cached(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存数据"""
        from backend.repositories.redis import get_redis
        
        try:
            redis = get_redis()
            ttl = ttl or self.default_ttl
            return await redis.set(key, value, ttl=ttl)
        except Exception as e:
            logger.warning("缓存写入失败", key=key, error=str(e))
            return False


class InMemoryRepository(BaseRepository[T, K]):
    """内存仓库实现 - 用于测试和简单场景"""
    
    def __init__(self):
        self._storage: Dict[K, T] = {}
        self._next_id = 1
    
    async def create(self, entity: T) -> T:
        """创建实体"""
        # 如果实体有id属性且为None，自动分配
        if hasattr(entity, 'id') and entity.id is None:
            entity.id = self._next_id
            self._next_id += 1
        
        entity_id = getattr(entity, 'id', self._next_id)
        self._storage[entity_id] = entity
        
        logger.debug("内存仓库创建实体", entity_id=entity_id)
        return entity
    
    async def get_by_id(self, entity_id: K) -> Optional[T]:
        """根据ID获取实体"""
        entity = self._storage.get(entity_id)
        logger.debug("内存仓库获取实体", entity_id=entity_id, found=entity is not None)
        return entity
    
    async def update(self, entity_id: K, updates: Dict[str, Any]) -> Optional[T]:
        """更新实体"""
        entity = self._storage.get(entity_id)
        if entity is None:
            return None
        
        # 更新属性
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        
        # 设置更新时间
        if hasattr(entity, 'updated_at'):
            entity.updated_at = datetime.now()
        
        logger.debug("内存仓库更新实体", entity_id=entity_id, updates=updates)
        return entity
    
    async def delete(self, entity_id: K) -> bool:
        """删除实体"""
        if entity_id in self._storage:
            del self._storage[entity_id]
            logger.debug("内存仓库删除实体", entity_id=entity_id)
            return True
        return False
    
    async def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """列出所有实体"""
        entities = list(self._storage.values())
        
        # 应用分页
        if offset > 0:
            entities = entities[offset:]
        if limit is not None:
            entities = entities[:limit]
        
        logger.debug("内存仓库列表查询", count=len(entities), total=len(self._storage))
        return entities
    
    async def count(self) -> int:
        """统计实体数量"""
        count = len(self._storage)
        logger.debug("内存仓库计数", count=count)
        return count
    
    async def clear(self):
        """清空所有数据"""
        self._storage.clear()
        self._next_id = 1
        logger.debug("内存仓库已清空")


class SearchableRepository(ABC, Generic[T]):
    """支持搜索的仓库接口"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[T]:
        """全文搜索"""
        pass
    
    @abstractmethod
    async def filter_by(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[T]:
        """根据条件过滤"""
        pass


class BulkRepository(ABC, Generic[T]):
    """支持批量操作的仓库接口"""
    
    @abstractmethod
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """批量创建"""
        pass
    
    @abstractmethod
    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """批量更新"""
        pass
    
    @abstractmethod
    async def bulk_delete(self, entity_ids: List[K]) -> int:
        """批量删除"""
        pass


class RepositoryFactory:
    """仓库工厂 - 用于创建和管理仓库实例"""
    
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_repository(cls, repo_class: type, **kwargs) -> Any:
        """获取仓库实例（单例）"""
        key = f"{repo_class.__name__}_{hash(frozenset(kwargs.items()))}"
        
        if key not in cls._instances:
            cls._instances[key] = repo_class(**kwargs)
            logger.debug("创建仓库实例", repo_class=repo_class.__name__)
        
        return cls._instances[key]
    
    @classmethod
    def clear_instances(cls):
        """清空所有实例（主要用于测试）"""
        cls._instances.clear()


# 常用的仓库装饰器
def cached_result(ttl: int = 3600):
    """缓存方法结果的装饰器"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # 生成缓存键
            cache_key = f"{self.__class__.__name__}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            try:
                # 尝试从缓存获取
                from backend.repositories.redis import get_redis
                redis = get_redis()
                cached_result = await redis.get(cache_key)
                
                if cached_result is not None:
                    logger.debug("缓存命中", method=func.__name__, cache_key=cache_key)
                    return cached_result
                
                # 执行原方法
                result = await func(self, *args, **kwargs)
                
                # 缓存结果
                await redis.set(cache_key, result, ttl=ttl)
                logger.debug("缓存结果", method=func.__name__, cache_key=cache_key)
                
                return result
                
            except Exception as e:
                logger.warning("缓存操作失败，直接执行方法", method=func.__name__, error=str(e))
                return await func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_entity(entity_class: type):
    """实体验证装饰器"""
    def decorator(func):
        async def wrapper(self, entity, *args, **kwargs):
            if not isinstance(entity, entity_class):
                raise ValueError(f"Expected {entity_class.__name__}, got {type(entity).__name__}")
            return await func(self, entity, *args, **kwargs)
        return wrapper
    return decorator