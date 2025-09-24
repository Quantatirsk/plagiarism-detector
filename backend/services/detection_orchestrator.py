"""Orchestration layer for document ingestion and pairwise comparison."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast, Any

from sqlmodel import select, col

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
from backend.services.progress_tracker import ProgressTracker, ProgressType
from backend.core.logging import get_logger, LogEvent

logger = get_logger(__name__)


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
        self.progress_tracker = ProgressTracker()

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

    async def ingest_files_batch(
        self,
        file_paths: List[str],
        *,
        project_id: int,
        titles: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[UploadedDocument]:
        """批量解析、分块和嵌入多个文档"""
        self._ensure_initialized()

        # Create parent task for batch processing
        parent_task_id = self.progress_tracker.create_task(
            task_type=ProgressType.BATCH_PROCESSING,
            description=f"Processing {len(file_paths)} documents",
            total_steps=len(file_paths),
            metadata={"project_id": project_id, "file_count": len(file_paths)}
        )
        await self.progress_tracker.start_task(parent_task_id)

        logger.info(
            "batch_processing_started",
            project_id=project_id,
            total_files=len(file_paths),
            progress_task_id=parent_task_id
        )

        # Convert None to list of None values to match expected types
        actual_titles: List[Optional[str]] = cast(List[Optional[str]], titles) if titles is not None else [None for _ in range(len(file_paths))]
        actual_sources: List[Optional[str]] = cast(List[Optional[str]], sources) if sources is not None else [None for _ in range(len(file_paths))]

        # 批量解析文档
        contents = []
        valid_indices = []
        for i, file_path in enumerate(file_paths):
            resolved_path = Path(file_path)
            if not resolved_path.exists():
                logger.warning(
                    "file_not_found_skipping",
                    file_path=file_path
                )
                continue

            content = self.parser.parse_document(str(resolved_path))
            if content is None:
                logger.warning(
                    "document_parse_failed_skipping",
                    file_path=file_path
                )
                continue

            contents.append(content)
            valid_indices.append(i)

        if not contents:
            return []

        # 批量处理文本
        logger.info(
            "batch_text_processing_started",
            document_count=len(contents)
        )

        await self.progress_tracker.update_progress(
            parent_task_id,
            message=f"Processing text for {len(contents)} documents"
        )

        processed_results = self.text_processor.process_documents_batch(contents)

        logger.info(
            "batch_text_processing_completed",
            document_count=len(processed_results)
        )

        # 处理每个有效文档
        uploaded_documents = []
        for idx, (content, result) in enumerate(zip(contents, processed_results)):
            original_idx = valid_indices[idx]
            file_path = file_paths[original_idx]
            resolved_path = Path(file_path)

            checksum = self._compute_checksum(resolved_path)
            language = self.text_processor.detect_language(content[:2000])
            metadata = {
                "source_path": str(resolved_path),
                "checksum": checksum,
            }

            upload = await self.register_upload(
                project_id=project_id,
                title=actual_titles[original_idx] or resolved_path.stem,
                filename=resolved_path.name,
                source=actual_sources[original_idx],
                checksum=checksum,
                processed_text=content,
                language=language,
                char_count=len(content),
                metadata=metadata,
                storage_path=str(resolved_path),
            )

            # Create sub-task for this document
            doc_task_id = self.progress_tracker.create_task(
                task_type=ProgressType.DOCUMENT_PROCESSING,
                description=f"Processing: {resolved_path.name}",
                parent_id=parent_task_id,
                metadata={"file_path": str(resolved_path), "document_id": upload.document.id}
            )
            await self.progress_tracker.start_task(doc_task_id)

            try:
                if upload.document.id is not None:
                    await self.mark_document_status(upload.document.id, status=DocumentStatus.PROCESSING)

                # 使用批处理结果
                paragraph_payloads = [
                    ChunkCreate(
                        chunk_type=ChunkGranularity.PARAGRAPH,
                        chunk_index=index,
                        start_pos=start,
                        end_pos=end,
                        parent_chunk_id=None,
                        text=segment,
                    )
                    for index, (segment, start, end) in enumerate(result['paragraphs'])
                ]

                if upload.document.id is None:
                    raise ValueError("Document ID is None")
                paragraphs = await self.persist_chunks(upload.document.id, paragraph_payloads)

                sentence_payloads: List[ChunkCreate] = []
                paragraph_lookup = self._build_parent_lookup(paragraphs)
                for index, (sentence_text, start, end) in enumerate(result['sentences'], start=len(paragraphs)):
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

                logger.info(
                    "document_batch_item_completed",
                    document_id=upload.document.id,
                    file_path=file_path,
                    paragraph_count=len(paragraphs),
                    sentence_count=len(sentence_chunks)
                )

                await self.progress_tracker.complete_task(
                    doc_task_id,
                    message=f"完成：{len(paragraphs)} 个段落，{len(sentence_chunks)} 个句子"
                )

                # Update parent progress
                await self.progress_tracker.update_progress(
                    parent_task_id,
                    current_step=len(uploaded_documents) + 1,
                    message=f"Processed {len(uploaded_documents) + 1}/{len(file_paths)} documents"
                )

                refreshed_document = await self.fetch_document(upload.document.id)
                uploaded_documents.append(UploadedDocument(document=refreshed_document or upload.document))

            except (EmbeddingError, StorageError, Exception) as exc:
                logger.error(
                    "document_batch_item_failed",
                    document_id=upload.document.id,
                    file_path=file_path,
                    error=str(exc),
                    exc_info=True
                )

                await self.progress_tracker.fail_task(
                    doc_task_id,
                    error_message=f"Failed: {str(exc)}"
                )
                if upload.document.id is not None:
                    await self.mark_document_status(
                        upload.document.id,
                        status=DocumentStatus.FAILED,
                        error_message=str(exc),
                    )

        logger.info(
            "batch_processing_completed",
            project_id=project_id,
            successful_count=len(uploaded_documents),
            total_count=len(file_paths),
            failed_count=len(file_paths) - len(uploaded_documents)
        )

        await self.progress_tracker.complete_task(
            parent_task_id,
            message=f"批处理完成：{len(uploaded_documents)} 个成功，{len(file_paths) - len(uploaded_documents)} 个失败"
        )
        return uploaded_documents

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

        # Create progress task
        task_id = self.progress_tracker.create_task(
            task_type=ProgressType.DOCUMENT_PROCESSING,
            description=f"Processing document: {resolved_path.name}",
            total_steps=6,  # 读取、解析、段落、句子、嵌入、完成
            metadata={"file_path": str(resolved_path), "project_id": project_id}
        )
        await self.progress_tracker.start_task(task_id)

        logger.info(
            LogEvent.DOCUMENT_PROCESSING_STARTED,
            file_path=str(resolved_path),
            project_id=project_id,
            title=title or resolved_path.stem,
            progress_task_id=task_id
        )

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

        logger.info(
            "document_parsed",
            file_path=str(resolved_path),
            char_count=len(content),
            language=language,
            checksum=checksum[:8]  # First 8 chars of checksum
        )

        await self.progress_tracker.update_progress(
            task_id,
            current_step=1,
            message=f"文档解析完成: {len(content)} 字符"
        )

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
            if upload.document.id is None:
                raise ValueError("Document ID is None")
            await self.mark_document_status(upload.document.id, status=DocumentStatus.PROCESSING)

            paragraph_segments = self.text_processor.split_paragraphs(content)
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

            logger.info(
                "paragraphs_created",
                document_id=upload.document.id,
                paragraph_count=len(paragraphs),
                avg_paragraph_length=sum(len(p.text) for p in paragraph_payloads) // len(paragraph_payloads) if paragraph_payloads else 0
            )

            await self.progress_tracker.update_progress(
                task_id,
                current_step=2,
                message=f"已创建 {len(paragraphs)} 个段落"
            )

            sentences = self.text_processor.split_sentences(content)
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

            logger.info(
                "sentences_created",
                document_id=upload.document.id,
                sentence_count=len(sentence_chunks),
                avg_sentence_length=sum(len(s.text) for s in sentence_payloads) // len(sentence_payloads) if sentence_payloads else 0
            )

            await self.progress_tracker.update_progress(
                task_id,
                current_step=3,
                message=f"已创建 {len(sentence_chunks)} 个句子，准备生成嵌入向量"
            )

            logger.info(
                "embedding_started",
                document_id=upload.document.id,
                sentence_count=len(sentence_chunks)
            )

            await self._embed_and_index(upload.document, sentence_chunks, task_id)

            logger.info(
                "embedding_completed",
                document_id=upload.document.id
            )

            await self.progress_tracker.update_progress(
                task_id,
                current_step=5,
                message="所有嵌入向量已生成并索引"
            )

            await self.mark_document_status(
                upload.document.id,
                status=DocumentStatus.COMPLETED,
                paragraph_count=len(paragraphs),
                sentence_count=len(sentence_chunks),
            )

            logger.info(
                LogEvent.DOCUMENT_PROCESSING_COMPLETED,
                document_id=upload.document.id,
                file_path=str(resolved_path),
                paragraph_count=len(paragraphs),
                sentence_count=len(sentence_chunks)
            )

            await self.progress_tracker.complete_task(
                task_id,
                message=f"文档处理成功：{len(paragraphs)} 个段落，{len(sentence_chunks)} 个句子"
            )
        except (EmbeddingError, StorageError, Exception) as exc:
            logger.error(
                LogEvent.DOCUMENT_PROCESSING_FAILED,
                document_id=upload.document.id,
                file_path=str(resolved_path),
                error=str(exc),
                exc_info=True
            )
            await self.progress_tracker.fail_task(
                task_id,
                error_message=f"Failed to process document: {str(exc)}"
            )
            if upload.document.id is not None:
                await self.mark_document_status(
                    upload.document.id,
                    status=DocumentStatus.FAILED,
                    error_message=str(exc),
                )
            raise

        refreshed_document = await self.fetch_document(upload.document.id)
        return UploadedDocument(document=refreshed_document or upload.document)

    async def process_existing_document(
        self,
        file_path: str,
        *,
        document_id: int,
        project_id: int,
    ) -> UploadedDocument:
        """Process a file for an existing document (update instead of create new)."""
        self._ensure_initialized()
        resolved_path = Path(file_path)
        if not resolved_path.exists():
            raise DocumentParseError("文件不存在", file_path=file_path)

        # Create progress task
        task_id = self.progress_tracker.create_task(
            task_type=ProgressType.DOCUMENT_PROCESSING,
            description=f"Processing document: {resolved_path.name}",
            total_steps=6,  # 读取、解析、段落、句子、嵌入、完成
            metadata={"file_path": str(resolved_path), "project_id": project_id, "document_id": document_id}
        )
        await self.progress_tracker.start_task(task_id)

        logger.info(
            LogEvent.DOCUMENT_PROCESSING_STARTED,
            file_path=str(resolved_path),
            project_id=project_id,
            document_id=document_id,
            progress_task_id=task_id
        )

        # Parse content
        content = self.parser.parse_document(str(resolved_path))
        if content is None:
            raise DocumentParseError("无法解析文档", file_path=file_path)

        checksum = self._compute_checksum(resolved_path)
        language = self.text_processor.detect_language(content[:2000])

        # Update existing document with parsed information
        await self.gateway.update_document(
            document_id,
            checksum=checksum,
            processed_text=content,
            language=language,
            char_count=len(content),
            filename=resolved_path.name,
            storage_path=str(resolved_path),
        )

        await self.progress_tracker.update_progress(
            task_id,
            current_step=1,
            message=f"文档解析完成: {len(content)} 字符"
        )

        # Get the updated document
        document = await self.fetch_document(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")

        try:
            await self.mark_document_status(document_id, status=DocumentStatus.PROCESSING)

            paragraph_segments = self.text_processor.split_paragraphs(content)
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
            paragraphs = await self.persist_chunks(document_id, paragraph_payloads)

            await self.progress_tracker.update_progress(
                task_id,
                current_step=2,
                message=f"已创建 {len(paragraphs)} 个段落"
            )

            sentence_segments = self.text_processor.split_sentences(content)
            sentence_payloads = []
            paragraph_lookup = self._build_parent_lookup(paragraphs)
            for index, (sentence_text, start, end) in enumerate(sentence_segments, start=len(paragraphs)):
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
            sentence_chunks = await self.persist_chunks(document_id, sentence_payloads)

            await self.progress_tracker.update_progress(
                task_id,
                current_step=3,
                message=f"已创建 {len(sentence_chunks)} 个句子，准备生成嵌入向量"
            )

            await self._embed_and_index(document, sentence_chunks, task_id)

            await self.progress_tracker.update_progress(
                task_id,
                current_step=5,
                message=f"已生成 {len(sentence_chunks)} 个文本块的嵌入向量"
            )

            await self.mark_document_status(
                document_id,
                status=DocumentStatus.COMPLETED,
                paragraph_count=len(paragraphs),
                sentence_count=len(sentence_chunks),
            )

            await self.progress_tracker.complete_task(
                task_id,
                message=f"文档处理成功：{len(paragraphs)} 个段落，{len(sentence_chunks)} 个句子"
            )

            logger.info(
                LogEvent.DOCUMENT_PROCESSING_COMPLETED,
                document_id=document_id,
                file_path=str(resolved_path),
                paragraph_count=len(paragraphs),
                sentence_count=len(sentence_chunks),
            )
        except (EmbeddingError, StorageError, Exception) as exc:
            logger.error(
                LogEvent.DOCUMENT_PROCESSING_FAILED,
                document_id=document_id,
                file_path=str(resolved_path),
                error=str(exc),
                exc_info=True
            )
            await self.progress_tracker.fail_task(
                task_id,
                error_message=f"Failed to process document: {str(exc)}"
            )
            await self.mark_document_status(
                document_id,
                status=DocumentStatus.FAILED,
                error_message=str(exc),
            )
            raise

        refreshed_document = await self.fetch_document(document_id)
        return UploadedDocument(document=refreshed_document or document)

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

        logger.info(
            "project_comparison_started",
            project_id=project_id
        )

        async with get_session() as session:
            stmt = (
                select(Document)
                .where(Document.project_id == project_id)
                .where(Document.status == DocumentStatus.COMPLETED)
                .order_by(col(Document.created_at).asc())
            )
            documents = (await session.exec(stmt)).all()

        if len(documents) < 2:
            logger.warning(
                "insufficient_documents_for_comparison",
                project_id=project_id,
                document_count=len(documents),
                required_count=2
            )
            return None

        logger.info(
            "documents_found_for_comparison",
            project_id=project_id,
            document_count=len(documents)
        )

        async with get_session() as session:
            job_stmt = (
                select(CompareJob)
                .where(CompareJob.project_id == project_id)
                .order_by(col(CompareJob.created_at).asc())
            )
            existing_job = (await session.exec(job_stmt)).first()

        job = existing_job
        if not job:
            job = await self.create_compare_job(project_id, f"Project {project_id} Comparison", None)

        if job.id is None:
            raise ValueError("Job ID is None")
        existing_pairs = await self.list_pairs_for_job(job.id)
        existing_keys = {
            tuple(sorted((pair.left_document_id, pair.right_document_id)))
            for pair in existing_pairs
            if pair.left_document_id is not None and pair.right_document_id is not None
        }

        document_ids = [document.id for document in documents if document.id is not None]
        new_pairs: List[tuple[int, int]] = []
        for index, left_id in enumerate(document_ids):
            for right_id in document_ids[index + 1 :]:
                key = (min(left_id, right_id), max(left_id, right_id))
                if key in existing_keys:
                    continue
                new_pairs.append((key[0], key[1]))

        if not new_pairs:
            logger.info(
                "no_new_pairs_to_compare",
                project_id=project_id,
                job_id=job.id,
                existing_pairs=len(existing_pairs)
            )
            await self.update_compare_job_status(job.id, CompareJobStatus.COMPLETED)
            return await self.fetch_job(job.id)

        logger.info(
            "comparison_job_starting",
            project_id=project_id,
            job_id=job.id,
            total_pairs=len(new_pairs),
            existing_pairs=len(existing_pairs)
        )

        # Create progress task for comparison job
        job_task_id = self.progress_tracker.create_task(
            task_type=ProgressType.COMPARISON_JOB,
            description=f"Comparing documents in project {project_id}",
            total_steps=len(new_pairs),
            metadata={"project_id": project_id, "job_id": job.id, "pair_count": len(new_pairs)}
        )
        await self.progress_tracker.start_task(job_task_id)

        await self.update_compare_job_status(job.id, CompareJobStatus.RUNNING)

        created_pairs = await self.add_pairs_to_job(job.id, new_pairs)

        from backend.services.comparison_service import ComparisonConfig, ComparisonService  # late import to avoid circular

        comparison_service = ComparisonService()
        config = ComparisonConfig()

        try:
            total_pairs = len(created_pairs)
            for index, pair in enumerate(created_pairs):
                logger.info(
                    "processing_pair",
                    job_id=job.id,
                    pair_id=pair.id,
                    current_pair=index + 1,
                    total_pairs=total_pairs,
                    progress_percent=round((index + 1) / total_pairs * 100, 2),
                    left_document_id=pair.left_document_id,
                    right_document_id=pair.right_document_id
                )

                # Create sub-task for this pair
                pair_task_id = self.progress_tracker.create_task(
                    task_type=ProgressType.COMPARISON_PAIR,
                    description=f"Comparing pair {index + 1}/{total_pairs}",
                    parent_id=job_task_id,
                    metadata={"pair_id": pair.id, "job_id": job.id}
                )
                await self.progress_tracker.start_task(pair_task_id)
                if pair.id is not None:
                    await comparison_service.run_pair(pair.id, config, progress_task_id=pair_task_id)

                await self.progress_tracker.complete_task(pair_task_id)

                # Update parent progress
                await self.progress_tracker.update_progress(
                    job_task_id,
                    current_step=index + 1,
                    message=f"Completed pair {index + 1}/{total_pairs}"
                )
        except Exception as e:
            logger.error(
                "comparison_job_failed",
                job_id=job.id,
                project_id=project_id,
                error=str(e),
                exc_info=True
            )
            await self.progress_tracker.fail_task(
                job_task_id,
                error_message=f"Comparison job failed: {str(e)}"
            )
            await self.update_compare_job_status(job.id, CompareJobStatus.FAILED)
            raise
        else:
            logger.info(
                "comparison_job_completed",
                job_id=job.id,
                project_id=project_id,
                total_pairs_processed=total_pairs
            )
            await self.progress_tracker.complete_task(
                job_task_id,
                message=f"比对完成：已处理 {total_pairs} 对文档"
            )
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
        pair_id: int,  # noqa: ARG002
        groups: Sequence[MatchGroupCreate],
        details: Sequence[MatchDetailCreate],
    ) -> None:
        self._ensure_initialized()
        stored_groups = await self.gateway.store_match_groups(groups)
        mapping: Dict[tuple[int, int], int] = {}
        for payload, stored in zip(groups, stored_groups):
            key = (payload.left_chunk_id, payload.right_chunk_id)
            if stored.id is not None:
                mapping[key] = stored.id

        normalised_details: List[MatchDetailCreate] = []
        for detail in details:
            if detail.group_id is not None and isinstance(detail.group_id, int):
                group_id = detail.group_id
            else:
                key = detail.group_key or detail.group_id
                try:
                    group_id = mapping.get(tuple(key)) if key else None  # type: ignore
                except TypeError:
                    group_id = None
            if group_id is None:
                continue
            normalised_details.append(
                MatchDetailCreate(
                    left_chunk_id=detail.left_chunk_id,
                    right_chunk_id=detail.right_chunk_id,
                    final_score=detail.final_score,
                    semantic_score=detail.semantic_score,
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
            stmt = stmt.order_by(col(Document.created_at).desc())
            result = await session.exec(stmt)
            return list(result.all())

    async def list_compare_jobs(self, project_id: Optional[int] = None) -> List[CompareJob]:
        self._ensure_initialized()
        async with get_session() as session:
            stmt = select(CompareJob)
            if project_id is not None:
                stmt = stmt.where(CompareJob.project_id == project_id)
            stmt = stmt.order_by(col(CompareJob.created_at).desc())
            result = await session.exec(stmt)
            return list(result.all())

    async def list_pairs_for_job(self, job_id: int) -> List[ComparePair]:
        self._ensure_initialized()
        async with get_session() as session:
            stmt = select(ComparePair).where(ComparePair.job_id == job_id)
            result = await session.exec(stmt)
            return list(result.all())

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
        task_id: Optional[str] = None,
    ) -> None:
        if not sentence_chunks:
            return

        texts = [chunk.text for chunk in sentence_chunks]
        total_chunks = len(texts)

        # 创建嵌入生成的子任务（如果有父任务）
        embed_task_id = None
        if task_id:
            embed_task_id = self.progress_tracker.create_task(
                task_type=ProgressType.EMBEDDING_GENERATION,
                description=f"生成 {total_chunks} 个文本块的嵌入向量",
                total_steps=total_chunks,
                parent_id=task_id,
                metadata={"document_id": document.id}
            )
            await self.progress_tracker.start_task(embed_task_id)

        # 进度回调函数
        async def embedding_progress_callback(progress_info: Dict[str, Any]) -> None:
            if embed_task_id:
                await self.progress_tracker.update_progress(
                    embed_task_id,
                    current_step=progress_info["current"],
                    progress_percent=progress_info["percent"],
                    message=progress_info["message"]
                )

        # 调用嵌入服务，传递进度回调
        embeddings = await self.embedding_service.embed_batch(texts, progress_callback=embedding_progress_callback)

        # 完成嵌入任务
        if embed_task_id:
            await self.progress_tracker.complete_task(embed_task_id, message="嵌入向量生成完成")

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
                chunk_id=chunk.id or 0,  # This should never be None for persisted chunks
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
