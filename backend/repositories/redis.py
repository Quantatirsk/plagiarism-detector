"""
Redis缓存服务 - 异步Redis操作，支持JSON序列化
"""
import json
import asyncio
from typing import Any, Optional, Union, Dict
from datetime import timedelta

import redis.asyncio as redis
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.errors import RedisError

logger = get_logger(__name__)
settings = get_settings()


class RedisRepository:
    """Redis缓存仓库 - 异步操作，支持JSON序列化和TTL管理"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self._client = redis_client
        self._connected = False
    
    @property
    def client(self) -> redis.Redis:
        """获取Redis客户端，延迟连接"""
        if self._client is None:
            self._client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
        return self._client
    
    async def connect(self) -> bool:
        """建立Redis连接"""
        try:
            await self.client.ping()
            self._connected = True
            logger.info("Redis连接成功", url=settings.redis_url)
            return True
        except Exception as e:
            logger.error("Redis连接失败", error=str(e))
            self._connected = False
            return False
    
    async def disconnect(self):
        """关闭Redis连接"""
        if self._client:
            await self._client.aclose()
            self._connected = False
            logger.info("Redis连接已关闭")
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        serialize: bool = True
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒或timedelta对象）
            serialize: 是否JSON序列化
        """
        try:
            # 处理TTL
            ex = None
            if ttl is not None:
                if isinstance(ttl, timedelta):
                    ex = int(ttl.total_seconds())
                else:
                    ex = ttl
            
            # 序列化值
            if serialize:
                cache_value = json.dumps(value, ensure_ascii=False)
            else:
                cache_value = str(value)
            
            # 设置缓存
            result = await self.client.set(key, cache_value, ex=ex)
            
            logger.debug("缓存设置成功", key=key, ttl=ex)
            return bool(result)
            
        except Exception as e:
            logger.error("缓存设置失败", key=key, error=str(e))
            raise RedisError(f"Failed to set cache: {e}", "set")
    
    async def get(
        self,
        key: str,
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            deserialize: 是否JSON反序列化
            default: 默认值（当键不存在时返回）
        """
        try:
            value = await self.client.get(key)
            
            if value is None:
                logger.debug("缓存未命中", key=key)
                return default
            
            # 反序列化
            if deserialize:
                try:
                    result = json.loads(value)
                except json.JSONDecodeError:
                    # 如果不是有效JSON，返回原始字符串
                    result = value
            else:
                result = value
            
            logger.debug("缓存命中", key=key)
            return result
            
        except Exception as e:
            logger.error("缓存获取失败", key=key, error=str(e))
            raise RedisError(f"Failed to get cache: {e}", "get")
    
    async def delete(self, key: str) -> bool:
        """删除缓存键"""
        try:
            result = await self.client.delete(key)
            logger.debug("缓存删除", key=key, deleted=bool(result))
            return bool(result)
            
        except Exception as e:
            logger.error("缓存删除失败", key=key, error=str(e))
            raise RedisError(f"Failed to delete cache: {e}", "delete")
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            result = await self.client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error("缓存检查失败", key=key, error=str(e))
            raise RedisError(f"Failed to check cache: {e}", "exists")
    
    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """设置键的过期时间"""
        try:
            if isinstance(ttl, timedelta):
                seconds = int(ttl.total_seconds())
            else:
                seconds = ttl
            
            result = await self.client.expire(key, seconds)
            logger.debug("缓存过期时间设置", key=key, ttl=seconds)
            return bool(result)
            
        except Exception as e:
            logger.error("缓存过期设置失败", key=key, error=str(e))
            raise RedisError(f"Failed to set expiration: {e}", "expire")
    
    async def ttl(self, key: str) -> int:
        """获取键的剩余生存时间（秒）"""
        try:
            result = await self.client.ttl(key)
            # -1: 永不过期, -2: 键不存在, >0: 剩余秒数
            return result
            
        except Exception as e:
            logger.error("缓存TTL查询失败", key=key, error=str(e))
            raise RedisError(f"Failed to get TTL: {e}", "ttl")
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """原子递增"""
        try:
            result = await self.client.incrby(key, amount)
            logger.debug("缓存递增", key=key, amount=amount, result=result)
            return result
            
        except Exception as e:
            logger.error("缓存递增失败", key=key, error=str(e))
            raise RedisError(f"Failed to increment: {e}", "incr")
    
    async def hset(self, key: str, field: str, value: Any, serialize: bool = True) -> bool:
        """设置哈希字段"""
        try:
            if serialize:
                cache_value = json.dumps(value, ensure_ascii=False)
            else:
                cache_value = str(value)
            
            result = await self.client.hset(key, field, cache_value)
            logger.debug("哈希字段设置", key=key, field=field)
            return bool(result)
            
        except Exception as e:
            logger.error("哈希设置失败", key=key, field=field, error=str(e))
            raise RedisError(f"Failed to set hash: {e}", "hset")
    
    async def hget(self, key: str, field: str, deserialize: bool = True, default: Any = None) -> Any:
        """获取哈希字段"""
        try:
            value = await self.client.hget(key, field)
            
            if value is None:
                return default
            
            if deserialize:
                try:
                    result = json.loads(value)
                except json.JSONDecodeError:
                    result = value
            else:
                result = value
            
            logger.debug("哈希字段获取", key=key, field=field)
            return result
            
        except Exception as e:
            logger.error("哈希获取失败", key=key, field=field, error=str(e))
            raise RedisError(f"Failed to get hash: {e}", "hget")
    
    async def hgetall(self, key: str, deserialize: bool = True) -> Dict[str, Any]:
        """获取哈希的所有字段"""
        try:
            result = await self.client.hgetall(key)
            
            if not result:
                return {}
            
            if deserialize:
                processed_result = {}
                for field, value in result.items():
                    try:
                        processed_result[field] = json.loads(value)
                    except json.JSONDecodeError:
                        processed_result[field] = value
                return processed_result
            
            return result
            
        except Exception as e:
            logger.error("哈希全量获取失败", key=key, error=str(e))
            raise RedisError(f"Failed to get all hash: {e}", "hgetall")
    
    async def clear_pattern(self, pattern: str) -> int:
        """根据模式删除键"""
        try:
            keys = await self.client.keys(pattern)
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info("批量删除缓存", pattern=pattern, count=deleted)
                return deleted
            return 0
            
        except Exception as e:
            logger.error("模式删除失败", pattern=pattern, error=str(e))
            raise RedisError(f"Failed to clear pattern: {e}", "clear_pattern")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基本操作
            test_key = "health_check"
            test_value = "ok"
            
            # 设置测试值
            await self.set(test_key, test_value, ttl=10, serialize=False)
            
            # 读取测试值
            result = await self.get(test_key, deserialize=False)
            
            # 删除测试值
            await self.delete(test_key)
            
            # 获取连接信息
            info = await self.client.info()
            
            return {
                "status": "healthy" if result == test_value else "degraded",
                "connected": self._connected,
                "test_passed": result == test_value,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except Exception as e:
            logger.error("Redis健康检查失败", error=str(e))
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


# 简化的缓存接口 - 遵循Linus原则
class RedisCache:
    """Redis缓存 - 简单有效"""
    
    def __init__(self):
        self.repo = RedisRepository()
        self.ttl = settings.redis_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            return await self.repo.get(key, deserialize=True, default=None)
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存"""
        try:
            return await self.repo.set(
                key,
                value,
                ttl=ttl or self.ttl,
                serialize=True
            )
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            return await self.repo.delete(key)
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.repo.exists(key)
        except Exception as e:
            logger.error("Cache exists check failed", key=key, error=str(e))
            return False


# 全局Redis实例
_redis_instance: Optional[RedisRepository] = None


def get_redis() -> RedisRepository:
    """获取Redis实例（单例）"""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisRepository()
    return _redis_instance


async def init_redis() -> RedisRepository:
    """初始化Redis连接"""
    redis_repo = get_redis()
    await redis_repo.connect()
    return redis_repo


async def close_redis():
    """关闭Redis连接"""
    global _redis_instance
    if _redis_instance:
        await _redis_instance.disconnect()
        _redis_instance = None