"""
嵌入服务 - OpenAI兼容的文本嵌入服务
"""
import asyncio
import hashlib
from typing import List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.errors import EmbeddingError
from backend.repositories.redis import RedisCache
from backend.services.base_service import BaseService, singleton


@singleton
class EmbeddingService(BaseService):
    """OpenAI嵌入服务 - 保持简单"""

    def _initialize(self):
        """初始化OpenAI客户端"""
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url
        )
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions
        self.batch_size = self.settings.embedding_batch_size
        self.cache: Optional[RedisCache] = RedisCache() if self.settings.redis_url else None
        self._cache_prefix = f"embedding:{self.model}:"
        self._cache_ready = False
        self._cache_lock: Optional[asyncio.Lock] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """单文本嵌入"""
        self._ensure_initialized()
        cache_key: Optional[str] = None
        cache_available = await self._ensure_cache_ready()
        if cache_available and self.cache is not None:
            cache_key = self._cache_key(text)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return [float(value) for value in cached]
        try:
            self.logger.info(f"Starting embedding for text of length {len(text)} characters")
            self.logger.info(f"Using model: {self.model}, API endpoint: {self.client.base_url}")

            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )

            vector_dimension = len(response.data[0].embedding)
            self.logger.info(f"Embedding completed successfully! Vector dimension: {vector_dimension}")
            embedding = response.data[0].embedding
            if cache_available and self.cache is not None and cache_key is not None:
                await self.cache.set(cache_key, embedding)
            return embedding
        except Exception as e:
            self.logger.error("Embedding failed", text_length=len(text), error=str(e))
            raise EmbeddingError(f"Failed to embed text: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入 - 使用asyncio并发处理单个文本嵌入"""
        import asyncio
        self._ensure_initialized()
        local_cache: dict[str, List[float]] = {}
        
        async def embed_single_safe(text: str) -> List[float]:
            """安全的单文本嵌入，失败时返回零向量"""
            cache_key = self._cache_key(text)
            if cache_key in local_cache:
                return local_cache[cache_key]
            try:
                result = await self.embed_text(text)
                local_cache[cache_key] = result
                return result
            except Exception as e:
                self.logger.error("Single embedding failed", text_length=len(text), error=str(e))
                return [0.0] * self.dimensions

        # 使用asyncio.gather并发处理所有文本
        self.logger.info(f"Starting batch embedding for {len(texts)} texts")
        embeddings = await asyncio.gather(*[embed_single_safe(text) for text in texts])

        self.logger.info(f"Batch embedding completed: {len(embeddings)} embeddings generated (concurrent processing)")
        return embeddings

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._cache_prefix}{digest}"

    async def _ensure_cache_ready(self) -> bool:
        if self.cache is None:
            return False
        if self._cache_ready:
            return True
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        async with self._cache_lock:
            if self._cache_ready or self.cache is None:
                return self._cache_ready
            try:
                connected = await self.cache.repo.connect()
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning("Disabling embedding cache due to Redis connection failure", error=str(exc))
                self.cache = None
                self._cache_ready = False
                return False
            if not connected:
                self.logger.warning("Disabling embedding cache; Redis connection unavailable")
                self.cache = None
                self._cache_ready = False
                return False
            self._cache_ready = True
            return True
