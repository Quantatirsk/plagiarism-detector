"""
嵌入服务 - OpenAI兼容的文本嵌入服务
"""
import openai
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from app.core.config import get_settings
from app.core.errors import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """OpenAI嵌入服务 - 保持简单"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.openai_model
        self.dimensions = settings.openai_dimensions
        self.batch_size = settings.openai_batch_size
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """单文本嵌入"""
        try:
            logger.info(f"Starting embedding for text of length {len(text)} characters")
            logger.info(f"Using model: {self.model}, API endpoint: {self.client.base_url}")

            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )

            vector_dimension = len(response.data[0].embedding)
            logger.info(f"Embedding completed successfully! Vector dimension: {vector_dimension}")
            return response.data[0].embedding
        except Exception as e:
            logger.error("Embedding failed", text_length=len(text), error=str(e))
            raise EmbeddingError(f"Failed to embed text: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入 - 使用asyncio并发处理单个文本嵌入"""
        import asyncio
        
        async def embed_single_safe(text: str) -> List[float]:
            """安全的单文本嵌入，失败时返回零向量"""
            try:
                return await self.embed_text(text)
            except Exception as e:
                logger.error("Single embedding failed", text_length=len(text), error=str(e))
                return [0.0] * self.dimensions
        
        # 使用asyncio.gather并发处理所有文本
        logger.info(f"Starting batch embedding for {len(texts)} texts")
        embeddings = await asyncio.gather(*[embed_single_safe(text) for text in texts])

        logger.info(f"Batch embedding completed: {len(embeddings)} embeddings generated (concurrent processing)")
        return embeddings


# 全局实例
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service