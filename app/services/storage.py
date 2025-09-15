"""
Milvus向量存储服务 - 支持本地和服务器两种模式
"""
from typing import List, Dict, Any, Optional
import numpy as np
from app.core.config import get_settings, MilvusMode
from app.core.errors import StorageError
from app.core.logging import get_logger
import uuid

logger = get_logger(__name__)
settings = get_settings()


class MilvusStorage:
    """Milvus向量存储 - 支持本地开发和生产模式"""
    
    def __init__(self):
        self.collection_name = settings.milvus_collection
        self.mode = settings.milvus_mode
        self.client = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """连接Milvus - 根据模式选择连接方式"""
        if self.mode == MilvusMode.LOCAL:
            self._connect_local()
        else:
            self._connect_server()
    
    def _connect_local(self):
        """本地模式 - 使用Milvus Lite文件数据库"""
        from pymilvus import MilvusClient
        
        # 根据官方文档，直接传入文件名即可创建本地数据库
        self.client = MilvusClient(settings.milvus_db_file)
        logger.info("Connected to Milvus Lite", db_file=settings.milvus_db_file)
        
        # 确保集合存在
        if not self.client.has_collection(self.collection_name):
            self._create_collection_local()
    
    def _connect_server(self):
        """生产模式 - 连接到Milvus服务器"""
        from pymilvus import Collection, utility, connections
        
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        logger.info("Connected to Milvus server",
                   host=settings.milvus_host,
                   port=settings.milvus_port)
        
        # 确保集合存在
        if not utility.has_collection(self.collection_name):
            self._create_collection_server()
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
    
    def _create_collection_local(self):
        """本地模式 - 创建集合"""
        # 根据官方文档，MilvusClient使用简化API
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=settings.openai_dimensions,
            metric_type="COSINE",  # 余弦相似度
            auto_id=True  # 让Milvus自动生成ID，我们用metadata存储自定义字段
        )
        logger.info("Created local collection", name=self.collection_name, dimension=settings.openai_dimensions)
    
    def _create_collection_server(self):
        """生产模式 - 创建集合（原始方法）"""
        from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=50000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.openai_dimensions),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="position", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "Document embeddings for plagiarism detection")
        collection = Collection(self.collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("embedding", index_params)
        logger.info("Created server collection with index", name=self.collection_name)
    
    async def insert_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """插入向量 - 支持两种模式"""
        try:
            if self.mode == MilvusMode.LOCAL:
                # 本地模式 - 使用MilvusClient的简化数据格式
                data = []
                for chunk, embedding in zip(chunks, embeddings):
                    # MilvusClient需要vector字段和可选的metadata
                    data.append({
                        "vector": embedding,
                        "document_id": chunk["document_id"],
                        "content": chunk["content"][:50000],  # 保留长内容，避免过长
                        "chunk_type": chunk.get("chunk_type", "paragraph"),
                        "position": chunk.get("position", 0)
                    })
                
                res = self.client.insert(
                    collection_name=self.collection_name,
                    data=data
                )
                
                logger.info("Inserted embeddings (local)", count=len(data))
                return len(data)
            
            else:
                # 生产模式 - 原始方法
                data = {
                    "id": [chunk["id"] for chunk in chunks],
                    "document_id": [chunk["document_id"] for chunk in chunks],
                    "content": [chunk["content"][:50000] for chunk in chunks],  # 匹配schema最大长度
                    "embedding": embeddings,
                    "chunk_type": [chunk["chunk_type"] for chunk in chunks],
                    "position": [chunk["position"] for chunk in chunks]
                }
                
                result = self.collection.insert(data)
                self.collection.flush()
                
                logger.info("Inserted embeddings (server)", count=len(chunks))
                return result.insert_count
                
        except Exception as e:
            logger.error("Failed to insert embeddings", mode=self.mode, error=str(e))
            raise StorageError(f"Insert failed: {e}", "insert")
    
    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 50,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量 - 支持两种模式"""
        try:
            if self.mode == MilvusMode.LOCAL:
                # 本地模式 - 使用MilvusClient简化API
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["document_id", "content", "chunk_type", "position"],
                    filter=filters if filters else None
                )
                
                matches = []
                for result in results[0]:  # 结果是嵌套列表
                    # 调试：检查Milvus返回的distance值
                    cosine_distance = result.get("distance", 0.0)
                    logger.info(f"Milvus LOCAL mode - Raw distance: {cosine_distance}")

                    # 修复：将余弦距离转换为余弦相似度
                    # 注意：确保相似度在合理范围内
                    cosine_similarity = 1.0 - cosine_distance
                    logger.info(f"Converted similarity: {cosine_similarity}")

                    matches.append({
                        "id": result.get("id"),
                        "document_id": result.get("document_id"),
                        "content": result.get("content"),
                        "chunk_type": result.get("chunk_type"),
                        "position": result.get("position"),
                        "similarity": cosine_similarity
                    })
                
                return matches
            
            else:
                # 生产模式 - 原始方法
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": max(64, top_k * 2)}
                }
                
                results = self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filters,
                    output_fields=["document_id", "content", "chunk_type", "position"]
                )
                
                matches = []
                for hits in results:
                    for hit in hits:
                        # 修复：将余弦距离转换为余弦相似度
                        cosine_distance = hit.distance
                        cosine_similarity = 1.0 - cosine_distance
                        matches.append({
                            "id": hit.id,
                            "document_id": hit.entity.get("document_id"),
                            "content": hit.entity.get("content"),
                            "chunk_type": hit.entity.get("chunk_type"),
                            "position": hit.entity.get("position"),
                            "similarity": cosine_similarity
                        })
                
                return matches
                
        except Exception as e:
            logger.error("Search failed", mode=self.mode, error=str(e))
            raise StorageError(f"Search failed: {e}", "search")
    
    async def delete_collection(self):
        """删除集合 - 用于测试清理"""
        try:
            if self.mode == MilvusMode.LOCAL:
                self.client.drop_collection(self.collection_name)
            else:
                from pymilvus import utility
                utility.drop_collection(self.collection_name)
            logger.info("Dropped collection", name=self.collection_name)
        except Exception as e:
            logger.error("Failed to drop collection", error=str(e))


# 全局实例
_storage_service = None

def get_storage_service() -> MilvusStorage:
    """获取存储服务实例"""
    global _storage_service
    if _storage_service is None:
        _storage_service = MilvusStorage()
    return _storage_service