"""
统一异步 HTTP 客户端管理器
提供全局单例的异步 HTTP 客户端，支持连接池复用
"""

import httpx
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from backend.core.logging import get_logger

logger = get_logger(__name__)


class AsyncHTTPClientManager:
    """异步 HTTP 客户端管理器（单例模式）"""

    _instance: Optional['AsyncHTTPClientManager'] = None
    _client: Optional[httpx.AsyncClient] = None
    _lock: Optional[asyncio.Lock] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化管理器"""
        if self._lock is None:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                # 如果没有事件循环，稍后创建
                pass

    async def _ensure_lock(self):
        """确保锁已创建"""
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def get_client(self) -> httpx.AsyncClient:
        """获取全局异步 HTTP 客户端"""
        await self._ensure_lock()
        assert self._lock is not None  # _ensure_lock() guarantees this

        if self._client is None:
            async with self._lock:
                if self._client is None:
                    # 创建客户端配置
                    # ⚠️ 注意：默认超时适用于普通 API 调用
                    # OCR 等长时间任务需要使用自定义超时配置
                    timeout = httpx.Timeout(
                        connect=10.0,  # 连接超时
                        read=300.0,    # 读取超时（提高到 300 秒以支持 OCR）
                        write=30.0,    # 写入超时
                        pool=5.0       # 连接池超时
                    )

                    limits = httpx.Limits(
                        max_connections=50,           # 最大连接数（单 worker）
                        max_keepalive_connections=20  # 保持活动连接数
                    )

                    self._client = httpx.AsyncClient(
                        timeout=timeout,
                        limits=limits,
                        follow_redirects=True,
                        http2=False  # 禁用 HTTP/2 (需要安装 h2 包)
                    )

                    logger.info("全局异步 HTTP 客户端已创建")

        return self._client

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("异步 HTTP 客户端已关闭")

    @asynccontextmanager
    async def request(
        self,
        method: str,
        url: str,
        custom_timeout: Optional[httpx.Timeout] = None,
        **kwargs
    ):
        """
        异步请求上下文管理器

        Args:
            method: HTTP 方法
            url: 请求 URL
            custom_timeout: 自定义超时配置（用于长时间任务如 OCR）
            **kwargs: 其他请求参数
        """
        client = await self.get_client()

        # 如果提供了自定义超时，覆盖默认超时
        if custom_timeout:
            kwargs['timeout'] = custom_timeout

        try:
            response = await client.request(method, url, **kwargs)
            yield response
        finally:
            # 响应会自动关闭，这里不需要额外处理
            pass


# 全局实例
_http_manager = AsyncHTTPClientManager()


async def get_async_client() -> httpx.AsyncClient:
    """获取全局异步 HTTP 客户端"""
    return await _http_manager.get_client()


async def close_async_client():
    """关闭全局异步 HTTP 客户端"""
    await _http_manager.close()
