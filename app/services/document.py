"""
文档服务 - 文档上传、处理和索引管理
"""
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from app.services.embedding import EmbeddingService
from app.services.storage import MilvusStorage
from app.services.text_processor import TextProcessor
from app.models.document import Document, DocumentChunk, ChunkType
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DocumentService:
    """文档服务 - 处理文档的上传和索引"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        storage: MilvusStorage,
        text_processor: TextProcessor
    ):
        self.embedding = embedding_service
        self.storage = storage
        self.processor = text_processor
    
    async def create_document(
        self,
        content: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """创建文档记录"""
        document_id = str(uuid.uuid4())
        
        # 创建文档实体
        document = Document(
            id=document_id,
            content=content,
            created_at=datetime.now(),
            chunks_count=0,
            status="pending",
            metadata=metadata or {}
        )
        
        if filename:
            document.metadata["filename"] = filename
        
        logger.info("Document created", document_id=document_id, filename=filename)
        return document
    
    async def process_document(
        self,
        document_id: str,
        content: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> int:
        """处理文档 - 分块、嵌入、存储"""
        try:
            logger.info("Processing document", document_id=document_id)
            
            # 1. 创建文本块
            chunks = self._create_chunks(
                document_id=document_id,
                content=content,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            if not chunks:
                logger.warning("No chunks created", document_id=document_id)
                return 0
            
            # 2. 生成嵌入向量
            texts = [chunk["content"] for chunk in chunks]
            embeddings = await self.embedding.embed_batch(texts)
            
            # 3. 存储到向量数据库
            inserted_count = await self.storage.insert_embeddings(
                chunks=chunks,
                embeddings=embeddings
            )
            
            logger.info(
                "Document processed successfully",
                document_id=document_id,
                chunks_count=inserted_count
            )
            
            return inserted_count
            
        except Exception as e:
            logger.error(
                "Failed to process document",
                document_id=document_id,
                error=str(e)
            )
            raise
    
    def _create_chunks(
        self,
        document_id: str,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """创建文档块"""
        chunks = []
        
        # 1. 段落级分块
        paragraphs = self.processor.split_paragraphs(content)
        for i, para in enumerate(paragraphs):
            chunk_id = f"{document_id}_para_{i}"
            chunks.append({
                "id": chunk_id,
                "document_id": document_id,
                "content": para,
                "chunk_type": ChunkType.PARAGRAPH.value,
                "position": i
            })
        
        # 2. 如果内容很长，使用滑动窗口分块
        if len(content) > chunk_size * 3:
            windows = self.processor.create_sliding_windows(
                content,
                window_size=chunk_size,
                overlap=overlap
            )
            for j, (window_text, pos) in enumerate(windows):
                chunk_id = f"{document_id}_win_{j}"
                chunks.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "content": window_text,
                    "chunk_type": ChunkType.PARAGRAPH.value,
                    "position": len(paragraphs) + j
                })
        
        return chunks
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档及其所有块"""
        try:
            # TODO: 实现从向量数据库删除文档的逻辑
            # 这需要Milvus支持按document_id过滤删除
            logger.info("Document deletion requested", document_id=document_id)
            return True
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            return False
    
    async def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """获取文档统计信息"""
        # TODO: 实现获取文档统计信息
        return {
            "document_id": document_id,
            "chunks_count": 0,
            "status": "unknown"
        }