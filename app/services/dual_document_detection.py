"""
双文档检测服务 - 纯计算版本（无向量数据库）
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from enum import Enum

from app.models.detection import SimilarityMatch
from app.core.errors import DocumentParseError, InvalidInputError
from app.services.base_service import BaseService, singleton


class ChunkType(str, Enum):
    """文本块类型"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@singleton
class DualDocumentDetectionService(BaseService):
    """双文档检测服务 - 纯计算版本"""

    def _initialize(self):
        """初始化服务依赖"""
        from app.services.service_factory import service_factory
        self.embedding = service_factory.get_embedding_service()
        self.processor = service_factory.get_text_processor()
        self.document_parser = service_factory.get_document_parser()

    async def compare_documents(
        self,
        doc1_path: str,
        doc2_path: str,
        granularity: str = "paragraph",
        threshold: Optional[float] = None,
        max_total_matches: Optional[int] = None
    ) -> Dict[str, Any]:
        """对比两个文档的相似度（只选一种粒度）"""
        self._ensure_initialized()
        start_time = datetime.now()
        task_id = str(uuid.uuid4())

        try:
            # 1. 解析文档内容
            doc1_content = self.document_parser.parse_document(doc1_path)
            doc2_content = self.document_parser.parse_document(doc2_path)

            if not doc1_content:
                raise DocumentParseError("无法解析第一个文档", file_path=doc1_path)
            if not doc2_content:
                raise DocumentParseError("无法解析第二个文档", file_path=doc2_path)

            # 2. 只执行一种粒度检测
            if granularity not in ("paragraph", "sentence"):
                raise InvalidInputError("granularity must be 'paragraph' or 'sentence'", field="granularity", value=granularity)

            chunk_type = ChunkType.PARAGRAPH if granularity == "paragraph" else ChunkType.SENTENCE
            self.logger.info(f"Processing with granularity={granularity}, chunk_type={chunk_type.value}")
            # 默认阈值按粒度选择
            from app.core.config import get_settings
            settings = get_settings()
            if threshold is None:
                threshold = settings.paragraph_similarity_threshold if chunk_type == ChunkType.PARAGRAPH else settings.sentence_similarity_threshold
            if max_total_matches is None:
                max_total_matches = settings.max_total_matches

            # 预先计算块与偏移范围（用于前端稳定定位）
            if chunk_type == ChunkType.PARAGRAPH:
                doc1_spans = self.processor.split_paragraphs_with_spans(doc1_content)
                doc2_spans = self.processor.split_paragraphs_with_spans(doc2_content)
                self.logger.info(f"Split into {len(doc1_spans)} paragraphs and {len(doc2_spans)} paragraphs")
            else:
                doc1_spans = self.processor.split_sentences_with_spans(doc1_content)
                doc2_spans = self.processor.split_sentences_with_spans(doc2_content)
                self.logger.info(f"Split into {len(doc1_spans)} sentences and {len(doc2_spans)} sentences")

            # 计算匹配（当前内部也会分割一次，为保证正确先修复功能，后续可去重优化）
            matches = await self._find_matches(
                doc1_content, doc2_content, chunk_type, threshold, max_total_matches
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # 3. 构建文档信息
            document1_info = self.document_parser.get_document_info(doc1_path)
            document1_info["content"] = doc1_content

            document2_info = self.document_parser.get_document_info(doc2_path)
            document2_info["content"] = doc2_content

            return {
                "task_id": task_id,
                "status": "completed",
                "granularity": granularity,
                "document1_info": document1_info,
                "document2_info": document2_info,
                "document1_spans": [
                    {"index": i, "start": s, "end": e}
                    for i, (_, s, e) in enumerate(doc1_spans)
                ],
                "document2_spans": [
                    {"index": i, "start": s, "end": e}
                    for i, (_, s, e) in enumerate(doc2_spans)
                ],
                "matches": matches,
                "processing_time": processing_time,
                "created_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Document comparison failed: {e}")
            raise

    async def _find_matches(
        self,
        doc1_content: str,
        doc2_content: str,
        chunk_type: ChunkType,
        threshold: float,
        max_total_matches: int
    ) -> List[SimilarityMatch]:
        """矩阵化计算两个文档的块相似度，并按阈值过滤"""

        # 分割文档并获取块与偏移
        if chunk_type == ChunkType.PARAGRAPH:
            doc1_spans = self.processor.split_paragraphs_with_spans(doc1_content)
            doc2_spans = self.processor.split_paragraphs_with_spans(doc2_content)
        else:
            doc1_spans = self.processor.split_sentences_with_spans(doc1_content)
            doc2_spans = self.processor.split_sentences_with_spans(doc2_content)

        doc1_chunks = [t for (t, _, __) in doc1_spans]
        doc2_chunks = [t for (t, _, __) in doc2_spans]

        if not doc1_chunks or not doc2_chunks:
            return []

        self.logger.info(f"matching: type={chunk_type.value}, doc1={len(doc1_chunks)} chunks, doc2={len(doc2_chunks)} chunks, threshold={threshold}")

        # 批量嵌入
        doc1_embeddings = await self.embedding.embed_batch(doc1_chunks)
        doc2_embeddings = await self.embedding.embed_batch(doc2_chunks)

        import numpy as np
        # 转为矩阵并做 L2 归一化
        E1 = np.array(doc1_embeddings, dtype=float)
        E2 = np.array(doc2_embeddings, dtype=float)
        if E1.size == 0 or E2.size == 0:
            return []

        def l2_normalize(mat: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

        E1n = l2_normalize(E1)
        E2n = l2_normalize(E2)

        # 相似度矩阵：余弦相似度
        S = E1n @ E2n.T  # shape (n1, n2)

        matches: List[SimilarityMatch] = []
        n1, n2 = S.shape

        # 句子级和段落级都使用贪心算法：全局最优一对一匹配
        # 收集所有满足阈值的候选匹配对
        candidates = []
        for i in range(n1):
            for j in range(n2):
                sim = float(S[i, j])
                if sim >= threshold:
                    candidates.append((i, j, sim))

        if not candidates:
            return []

        # 按相似度降序排序
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 贪心算法：选择最优匹配对，确保一对一
        used_a = set()
        used_b = set()

        for i, j, sim in candidates:
            # 如果双方都未被匹配，则选择这对
            if i not in used_a and j not in used_b:
                matches.append(SimilarityMatch(
                    query_text=doc1_chunks[i],
                    matched_text=doc2_chunks[j],
                    similarity_score=sim,
                    document_id=f"doc2_{chunk_type.value}",
                    query_document_id=f"doc1_{chunk_type.value}",
                    position=int(j),
                    query_index=int(i),
                    match_index=int(j)
                ))
                used_a.add(i)
                used_b.add(j)

                # 达到最大匹配数限制
                if len(matches) >= max_total_matches:
                    break

        # 按相似度降序返回
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches

    # 已用矩阵化方式计算余弦相似度
