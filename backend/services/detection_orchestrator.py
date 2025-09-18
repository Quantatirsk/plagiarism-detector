"""Orchestration layer for document ingestion and pairwise comparison."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlmodel import select

from backend.core.errors import DocumentParseError, EmbeddingError, StorageError
from backend.db import get_session
from backend.db.models import (
    ChunkGranularity,
    CompareJob,
    CompareJobStatus,
    ComparePair,
    ComparePairStatus,
    Document,
    DocumentChunk,
    DocumentStatus,
    MatchDetail,
    MatchGroup,
    Project,
)
from backend.services.base_service import BaseService, singleton
from backend.services.document_parser import DocumentParser
from backend.services.storage_gateway import (
    ChunkCreate,
    CompareJobCreate,
    ComparePairCreate,
    DocumentCreate,
    EmbeddingCreate,
    MatchDetailCreate,
    MatchGroupCreate,
    ProjectCreate,
    StorageGateway,
)
from backend.services.text_processor import TextProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_storage import MilvusStorage


@dataclass
class UploadedDocument:
    document: Document


@dataclass
class CompareReport:
    pair: ComparePair
    left_document: Document
    right_document: Document
    groups: List[MatchGroup]
    details: List[MatchDetail]


@singleton
class DetectionOrchestrator(BaseService):
    """Facade coordinating document uploads and comparison jobs."""

    def _initialize(self) -> None:
        self.gateway = StorageGateway()
        self.parser = DocumentParser()
        self.text_processor = TextProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_storage = MilvusStorage()

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    async def create_project(self, name: Optional[str], description: Optional[str]) -> Project:
        self._ensure_initialized()
        return await self.gateway.create_project(ProjectCreate(name=name, description=description))

    async def list_projects(self) -> List[Project]:
        self._ensure_initialized()
        return await self.gateway.list_projects()

    async def fetch_project(self, project_id: int) -> Optional[Project]:
        self._ensure_initialized()
        return await self.gateway.fetch_project(project_id)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    async def register_upload(
        self,
        *,
        project_id: int,
        title: Optional[str],
        filename: Optional[str],
        source: Optional[str],
        checksum: str,
        processed_text: Optional[str],
        language: Optional[str],
        char_count: int,
        metadata: Optional[dict],
        storage_path: Optional[str],
    ) -> UploadedDocument:
        """Create document + version records for an uploaded file."""
        self._ensure_initialized()

        document = await self.gateway.create_document(
            DocumentCreate(
                project_id=project_id,
                title=title,
                filename=filename,
                source=source,
                checksum=checksum,
                storage_path=storage_path,
                processed_text=processed_text,
                language=language,
                char_count=char_count,
                metadata=metadata,
            )
        )
        return UploadedDocument(document=document)

    async def ingest_file(
        self,
        file_path: str,
        *,
        project_id: int,
        title: Optional[str] = None,
        source: Optional[str] = None,
    ) -> UploadedDocument:
        """Parse, chunk, and embed a document located at file_path."""

        self._ensure_initialized()
        resolved_path = Path(file_path)
        if not resolved_path.exists():
            raise DocumentParseError("文件不存在", file_path=file_path)

        # Parse content
        content = self.parser.parse_document(str(resolved_path))
        if content is None:
            raise DocumentParseError("无法解析文档", file_path=file_path)

        checksum = self._compute_checksum(resolved_path)
        language = self.text_processor.detect_language(content[:2000])
        metadata = {
            "source_path": str(resolved_path),
            "checksum": checksum,
        }

        upload = await self.register_upload(
            project_id=project_id,
            title=title or resolved_path.stem,
            filename=resolved_path.name,
            source=source,
            checksum=checksum,
            processed_text=content,
            language=language,
            char_count=len(content),
            metadata=metadata,
            storage_path=str(resolved_path),
        )

        try:
            await self.mark_document_status(upload.document.id, status=DocumentStatus.PROCESSING)

            paragraph_segments = self.text_processor.split_paragraphs_with_spans(content)
            paragraph_payloads = [
                ChunkCreate(
                    chunk_type=ChunkGranularity.PARAGRAPH,
                    chunk_index=index,
                    start_pos=start,
                    end_pos=end,
                    parent_chunk_id=None,
                    text=segment,
                )
                for index, (segment, start, end) in enumerate(paragraph_segments)
            ]

            paragraphs = await self.persist_chunks(upload.document.id, paragraph_payloads)

            sentences = self.text_processor.split_sentences_with_spans(content)
            sentence_payloads: List[ChunkCreate] = []
            paragraph_lookup = self._build_parent_lookup(paragraphs)
            for index, (sentence_text, start, end) in enumerate(sentences, start=len(paragraphs)):
                parent_chunk_id = self._locate_parent_chunk(start, end, paragraph_lookup)
                sentence_payloads.append(
                    ChunkCreate(
                        chunk_type=ChunkGranularity.SENTENCE,
                        chunk_index=index,
                        start_pos=start,
                        end_pos=end,
                        parent_chunk_id=parent_chunk_id,
                        text=sentence_text,
                    )
                )

            sentence_chunks = await self.persist_chunks(upload.document.id, sentence_payloads)

            await self._embed_and_index(upload.document, sentence_chunks)

            await self.mark_document_status(
                upload.document.id,
                status=DocumentStatus.COMPLETED,
                paragraph_count=len(paragraphs),
                sentence_count=len(sentence_chunks),
            )
        except (EmbeddingError, StorageError, Exception) as exc:
            await self.mark_document_status(
                upload.document.id,
                status=DocumentStatus.FAILED,
                error_message=str(exc),
            )
            raise

        refreshed_document = await self.fetch_document(upload.document.id)
        return UploadedDocument(document=refreshed_document or upload.document)

    async def mark_document_status(
        self,
        document_id: int,
        *,
        status: DocumentStatus,
        paragraph_count: Optional[int] = None,
        sentence_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Document:
        """Update processing status for a document."""
        self._ensure_initialized()
        return await self.gateway.update_document_status(
            document_id,
            status=status,
            paragraph_count=paragraph_count,
            sentence_count=sentence_count,
            error_message=error_message,
        )

    async def persist_chunks(
        self,
        version_id: int,
        chunks: Sequence[ChunkCreate],
    ) -> List[DocumentChunk]:
        """Store chunk metadata for a processed document version."""
        self._ensure_initialized()
        return await self.gateway.store_chunks(version_id, chunks)

    async def persist_embeddings(self, embeddings: Sequence[EmbeddingCreate]) -> None:
        self._ensure_initialized()
        await self.gateway.store_embeddings(embeddings)

    # ------------------------------------------------------------------
    # Comparison jobs
    # ------------------------------------------------------------------

    async def create_compare_job(self, project_id: int, name: Optional[str], config: Optional[dict]) -> CompareJob:
        self._ensure_initialized()
        return await self.gateway.create_compare_job(
            CompareJobCreate(project_id=project_id, name=name, config=config)
        )

    async def run_project_comparisons(self, project_id: int) -> Optional[CompareJob]:
        self._ensure_initialized()

        async with get_session() as session:
            stmt = (
                select(Document)
                .where(Document.project_id == project_id)
                .where(Document.status == DocumentStatus.COMPLETED)
                .order_by(Document.created_at.asc())
            )
            documents = (await session.exec(stmt)).all()

        if len(documents) < 2:
            return None

        async with get_session() as session:
            job_stmt = (
                select(CompareJob)
                .where(CompareJob.project_id == project_id)
                .order_by(CompareJob.created_at.asc())
            )
            existing_job = (await session.exec(job_stmt)).first()

        job = existing_job
        if not job:
            job = await self.create_compare_job(project_id, f"Project {project_id} Comparison", None)

        existing_pairs = await self.list_pairs_for_job(job.id)
        existing_keys = {
            tuple(sorted((pair.left_document_id, pair.right_document_id)))
            for pair in existing_pairs
        }

        document_ids = [document.id for document in documents]
        new_pairs: List[tuple[int, int]] = []
        for index, left_id in enumerate(document_ids):
            for right_id in document_ids[index + 1 :]:
                key = tuple(sorted((left_id, right_id)))
                if key in existing_keys:
                    continue
                new_pairs.append((key[0], key[1]))

        if not new_pairs:
            await self.update_compare_job_status(job.id, CompareJobStatus.COMPLETED)
            return await self.fetch_job(job.id)

        await self.update_compare_job_status(job.id, CompareJobStatus.RUNNING)

        created_pairs = await self.add_pairs_to_job(job.id, new_pairs)

        from backend.services.comparison_service import ComparisonConfig, ComparisonService  # late import to avoid circular

        comparison_service = ComparisonService()
        config = ComparisonConfig()

        try:
            for pair in created_pairs:
                await comparison_service.run_pair(pair.id, config)
        except Exception:
            await self.update_compare_job_status(job.id, CompareJobStatus.FAILED)
            raise
        else:
            await self.update_compare_job_status(job.id, CompareJobStatus.COMPLETED)

        return await self.fetch_job(job.id)

    async def update_compare_job_status(self, job_id: int, status: CompareJobStatus) -> CompareJob:
        self._ensure_initialized()
        return await self.gateway.update_compare_job_status(job_id, status=status)

    async def add_pairs_to_job(
        self,
        job_id: int,
        pairs: Sequence[tuple[int, int]],
    ) -> List[ComparePair]:
        self._ensure_initialized()
        payloads = [
            ComparePairCreate(job_id=job_id, left_document_id=left, right_document_id=right)
            for left, right in pairs
        ]
        return await self.gateway.add_pairs(payloads)

    async def update_pair_status(
        self,
        pair_id: int,
        *,
        status: ComparePairStatus,
        metrics: Optional[dict] = None,
    ) -> ComparePair:
        self._ensure_initialized()
        return await self.gateway.update_pair_status(pair_id, status=status, metrics=metrics)

    async def persist_match_results(
        self,
        pair_id: int,
        groups: Sequence[MatchGroupCreate],
        details: Sequence[MatchDetailCreate],
    ) -> None:
        self._ensure_initialized()
        stored_groups = await self.gateway.store_match_groups(groups)
        mapping: Dict[tuple[int, int], int] = {}
        for payload, stored in zip(groups, stored_groups):
            key = (payload.left_chunk_id, payload.right_chunk_id)
            mapping[key] = stored.id

        normalised_details: List[MatchDetailCreate] = []
        for detail in details:
            if detail.group_id is not None and isinstance(detail.group_id, int):
                group_id = detail.group_id
            else:
                key = detail.group_key or detail.group_id
                group_id = mapping.get(tuple(key)) if key else None
            if group_id is None:
                continue
            normalised_details.append(
                MatchDetailCreate(
                    left_chunk_id=detail.left_chunk_id,
                    right_chunk_id=detail.right_chunk_id,
                    final_score=detail.final_score,
                    semantic_score=detail.semantic_score,
                    lexical_overlap=detail.lexical_overlap,
                    cross_score=detail.cross_score,
                    spans=detail.spans,
                    group_id=group_id,
                )
            )

        await self.gateway.store_match_details(normalised_details)

    async def fetch_compare_report(self, pair_id: int) -> CompareReport:
        self._ensure_initialized()
        report = await self.gateway.fetch_match_report(pair_id)
        return CompareReport(
            pair=report["pair"],
            left_document=report["left_document"],
            right_document=report["right_document"],
            groups=[item["group"] for item in report["groups"]],
            details=[detail for item in report["groups"] for detail in item["details"]],
        )

    async def list_documents(
        self,
        status: Optional[DocumentStatus] = None,
        project_id: Optional[int] = None,
    ) -> List[Document]:
        self._ensure_initialized()
        async with get_session() as session:
            stmt = select(Document)
            if project_id is not None:
                stmt = stmt.where(Document.project_id == project_id)
            if status is not None:
                stmt = stmt.where(Document.status == status)
            stmt = stmt.order_by(Document.created_at.desc())
            result = await session.exec(stmt)
            return result.all()

    async def list_compare_jobs(self, project_id: Optional[int] = None) -> List[CompareJob]:
        self._ensure_initialized()
        async with get_session() as session:
            stmt = select(CompareJob)
            if project_id is not None:
                stmt = stmt.where(CompareJob.project_id == project_id)
            stmt = stmt.order_by(CompareJob.created_at.desc())
            result = await session.exec(stmt)
            return result.all()

    async def list_pairs_for_job(self, job_id: int) -> List[ComparePair]:
        self._ensure_initialized()
        async with get_session() as session:
            stmt = select(ComparePair).where(ComparePair.job_id == job_id)
            result = await session.exec(stmt)
            return result.all()

    async def fetch_job(self, job_id: int) -> Optional[CompareJob]:
        self._ensure_initialized()
        return await self.gateway.fetch_compare_job(job_id)

    async def fetch_pair(self, pair_id: int) -> Optional[ComparePair]:
        self._ensure_initialized()
        return await self.gateway.fetch_pair(pair_id)

    async def fetch_chunks(
        self,
        document_id: int,
        chunk_type: Optional[ChunkGranularity] = None,
    ) -> List[DocumentChunk]:
        self._ensure_initialized()
        return await self.gateway.fetch_chunks_by_document(document_id, chunk_type=chunk_type)

    async def fetch_document(self, document_id: int) -> Optional[Document]:
        self._ensure_initialized()
        async with get_session() as session:
            return await session.get(Document, document_id)

    async def delete_documents(self, document_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        await self.gateway.delete_documents(document_ids)

    async def delete_pairs(self, pair_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        await self.gateway.delete_pairs(pair_ids)

    async def delete_job(self, job_id: int) -> None:
        self._ensure_initialized()
        await self.gateway.delete_job(job_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_checksum(self, path: Path) -> str:
        buffer = path.read_bytes()
        return sha256(buffer).hexdigest()

    def _build_parent_lookup(
        self,
        paragraphs: Sequence[DocumentChunk],
    ) -> List[tuple[int, int, DocumentChunk]]:
        return [
            (paragraph.start_pos, paragraph.end_pos, paragraph)
            for paragraph in sorted(paragraphs, key=lambda item: item.start_pos)
        ]

    def _locate_parent_chunk(
        self,
        start: int,
        end: int,
        lookup: Sequence[tuple[int, int, DocumentChunk]],
    ) -> Optional[int]:
        for left, right, chunk in lookup:
            if start >= left and end <= right:
                return chunk.id
        if lookup:
            # fallback to closest chunk based on start offset
            _, _, chunk = min(lookup, key=lambda item: abs(item[0] - start))
            return chunk.id
        return None

    async def _embed_and_index(
        self,
        document: Document,
        sentence_chunks: Sequence[DocumentChunk],
    ) -> None:
        if not sentence_chunks:
            return

        texts = [chunk.text for chunk in sentence_chunks]
        embeddings = await self.embedding_service.embed_batch(texts)

        metadata = [
            {
                "id": str(chunk.id),
                "document_id": str(document.id),
                "content": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "position": chunk.chunk_index,
            }
            for chunk in sentence_chunks
        ]

        await self.vector_storage.insert_embeddings(metadata, embeddings)

        embedding_payloads = [
            EmbeddingCreate(
                chunk_id=chunk.id,
                vector_id=str(chunk.id),
                model=self.embedding_service.model,
                dimension=len(vector),
                norm=self._vector_norm(vector),
            )
            for chunk, vector in zip(sentence_chunks, embeddings)
        ]
        await self.persist_embeddings(embedding_payloads)

    def _vector_norm(self, embedding: Sequence[float]) -> float:
        return float(sum(value * value for value in embedding) ** 0.5)
