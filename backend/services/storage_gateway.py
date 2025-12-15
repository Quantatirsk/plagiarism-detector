"""Storage gateway for document library and comparison workflow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, or_
from sqlalchemy.exc import NoResultFound
from sqlmodel import col, select

from db import get_session
from db.models import (
    ChunkEmbedding,
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
from services.base_service import BaseService, singleton

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# ---------------------------------------------------------------------------
# Data payloads
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProjectCreate:
    name: str | None
    description: str | None


@dataclass(slots=True)
class DocumentCreate:
    project_id: int
    title: str | None
    filename: str | None
    source: str | None
    checksum: str
    storage_path: str | None
    processed_text: str | None
    language: str | None
    char_count: int
    metadata: dict | None


@dataclass(slots=True)
class ChunkCreate:
    chunk_type: ChunkGranularity
    chunk_index: int
    start_pos: int
    end_pos: int
    parent_chunk_id: int | None
    text: str
    text_hash: str | None = None


@dataclass(slots=True)
class EmbeddingCreate:
    chunk_id: int
    vector_id: str
    model: str
    dimension: int
    norm: float | None


@dataclass(slots=True)
class CompareJobCreate:
    project_id: int
    name: str | None
    config: dict | None


@dataclass(slots=True)
class ComparePairCreate:
    job_id: int
    left_document_id: int
    right_document_id: int


@dataclass(slots=True)
class MatchGroupCreate:
    pair_id: int
    left_chunk_id: int
    right_chunk_id: int
    final_score: float | None
    semantic_score: float | None
    cross_score: float | None
    alignment_ratio: float | None
    span_count: int
    match_count: int
    paragraph_spans: list[dict] | None
    document_spans: list[dict] | None


@dataclass(slots=True)
class MatchDetailCreate:
    left_chunk_id: int
    right_chunk_id: int
    final_score: float | None
    semantic_score: float | None
    cross_score: float | None
    spans: list[dict] | None
    group_id: int | None = None
    group_key: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Gateway implementation
# ---------------------------------------------------------------------------


@singleton
class StorageGateway(BaseService):
    """Coordinates relational persistence for document library workflow."""

    async def create_project(self, payload: ProjectCreate) -> Project:
        async with get_session() as session:
            project = Project(
                name=payload.name,
                description=payload.description,
            )
            session.add(project)
            await session.flush()
            return project

    async def list_projects(self) -> list[Project]:
        async with get_session() as session:
            stmt = select(Project).order_by(col(Project.created_at).desc())
            result = await session.exec(stmt)
            return list(result.all())

    async def fetch_project(self, project_id: int) -> Project | None:
        async with get_session() as session:
            return await session.get(Project, project_id)

    async def create_document(self, payload: DocumentCreate) -> Document:
        async with get_session() as session:
            doc = Document(
                project_id=payload.project_id,
                title=payload.title,
                filename=payload.filename,
                source=payload.source,
                checksum=payload.checksum,
                status=DocumentStatus.PENDING,
                language=payload.language,
                char_count=payload.char_count,
                processed_text=payload.processed_text,
                storage_path=payload.storage_path,
                metadata_json=payload.metadata,
            )
            session.add(doc)
            await session.flush()
            return doc

    async def update_document_status(
        self,
        document_id: int,
        *,
        status: DocumentStatus,
        paragraph_count: int | None = None,
        sentence_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        async with get_session() as session:
            document = await session.get(Document, document_id)
            if not document:
                raise NoResultFound(f"Document {document_id} not found")

            document.status = status
            document.updated_at = datetime.utcnow()
            if status == DocumentStatus.COMPLETED:
                document.completed_at = datetime.utcnow()
            if paragraph_count is not None:
                document.paragraph_count = paragraph_count
            if sentence_count is not None:
                document.sentence_count = sentence_count
            if error_message is not None:
                document.error_message = error_message
            await session.flush()
            return document

    async def update_document(
        self,
        document_id: int,
        *,
        checksum: str | None = None,
        processed_text: str | None = None,
        language: str | None = None,
        char_count: int | None = None,
        filename: str | None = None,
        storage_path: str | None = None,
    ) -> Document:
        """Update document with parsed information."""
        async with get_session() as session:
            document = await session.get(Document, document_id)
            if not document:
                raise NoResultFound(f"Document {document_id} not found")

            if checksum is not None:
                document.checksum = checksum
            if processed_text is not None:
                document.processed_text = processed_text
            if language is not None:
                document.language = language
            if char_count is not None:
                document.char_count = char_count
            if filename is not None:
                document.filename = filename
            if storage_path is not None:
                document.storage_path = storage_path

            document.updated_at = datetime.utcnow()

            await session.commit()
            await session.refresh(document)
            return document

    async def store_chunks(
        self,
        document_id: int,
        chunks: Sequence[ChunkCreate],
    ) -> list[DocumentChunk]:
        async with get_session() as session:
            stored: list[DocumentChunk] = []
            for entry in chunks:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_type=entry.chunk_type,
                    chunk_index=entry.chunk_index,
                    start_pos=entry.start_pos,
                    end_pos=entry.end_pos,
                    parent_chunk_id=entry.parent_chunk_id,
                    text=entry.text,
                    text_hash=entry.text_hash,
                )
                session.add(chunk)
                stored.append(chunk)
            await session.flush()
            return stored

    async def store_embeddings(self, embeddings: Sequence[EmbeddingCreate]) -> None:
        if not embeddings:
            return
        async with get_session() as session:
            for entry in embeddings:
                record = ChunkEmbedding(
                    chunk_id=entry.chunk_id,
                    vector_id=entry.vector_id,
                    model=entry.model,
                    dimension=entry.dimension,
                    norm=entry.norm,
                )
                session.add(record)
            await session.flush()

    async def create_compare_job(self, payload: CompareJobCreate) -> CompareJob:
        async with get_session() as session:
            job = CompareJob(
                project_id=payload.project_id,
                name=payload.name,
                config_json=payload.config,
                status=CompareJobStatus.DRAFT,
            )
            session.add(job)
            await session.flush()
            return job

    async def update_compare_job_status(
        self,
        job_id: int,
        *,
        status: CompareJobStatus,
    ) -> CompareJob:
        async with get_session() as session:
            job = await session.get(CompareJob, job_id)
            if not job:
                raise NoResultFound(f"CompareJob {job_id} not found")
            job.status = status
            job.updated_at = datetime.utcnow()
            if status == CompareJobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.utcnow()
            if status in {CompareJobStatus.COMPLETED, CompareJobStatus.FAILED}:
                job.completed_at = datetime.utcnow()
            await session.flush()
            return job

    async def add_pairs(self, pairs: Sequence[ComparePairCreate]) -> list[ComparePair]:
        async with get_session() as session:
            stored: list[ComparePair] = []
            for entry in pairs:
                pair = ComparePair(
                    job_id=entry.job_id,
                    left_document_id=entry.left_document_id,
                    right_document_id=entry.right_document_id,
                    status=ComparePairStatus.PENDING,
                )
                session.add(pair)
                stored.append(pair)
            await session.flush()
            return stored

    async def update_pair_status(
        self,
        pair_id: int,
        *,
        status: ComparePairStatus,
        metrics: dict | None = None,
    ) -> ComparePair:
        async with get_session() as session:
            pair = await session.get(ComparePair, pair_id)
            if not pair:
                raise NoResultFound(f"ComparePair {pair_id} not found")
            if status == ComparePairStatus.RUNNING and pair.started_at is None:
                pair.started_at = datetime.utcnow()
            if status in {ComparePairStatus.COMPLETED, ComparePairStatus.FAILED, ComparePairStatus.SKIPPED}:
                pair.completed_at = datetime.utcnow()
            pair.status = status
            if metrics is not None:
                pair.metrics_json = metrics
            pair.updated_at = datetime.utcnow()
            await session.flush()
            return pair

    async def store_match_groups(
        self,
        groups: Sequence[MatchGroupCreate],
    ) -> list[MatchGroup]:
        async with get_session() as session:
            stored: list[MatchGroup] = []
            for entry in groups:
                record = MatchGroup(
                    pair_id=entry.pair_id,
                    left_chunk_id=entry.left_chunk_id,
                    right_chunk_id=entry.right_chunk_id,
                    final_score=entry.final_score,
                    semantic_score=entry.semantic_score,
                    cross_score=entry.cross_score,
                    alignment_ratio=entry.alignment_ratio,
                    span_count=entry.span_count,
                    match_count=entry.match_count,
                    paragraph_spans_json=entry.paragraph_spans,
                    document_spans_json=entry.document_spans,
                )
                session.add(record)
                stored.append(record)
            await session.flush()
            return stored

    async def store_match_details(
        self,
        details: Sequence[MatchDetailCreate],
    ) -> None:
        if not details:
            return
        async with get_session() as session:
            for entry in details:
                if entry.group_id is None:
                    raise ValueError(f"MatchDetail requires group_id, got None for chunks {entry.left_chunk_id}/{entry.right_chunk_id}")
                record = MatchDetail(
                    group_id=entry.group_id,
                    left_chunk_id=entry.left_chunk_id,
                    right_chunk_id=entry.right_chunk_id,
                    final_score=entry.final_score,
                    semantic_score=entry.semantic_score,
                    cross_score=entry.cross_score,
                    spans_json=entry.spans,
                )
                session.add(record)
            await session.flush()

    async def fetch_chunks_by_document(
        self,
        document_id: int,
        *,
        chunk_type: ChunkGranularity | None = None,
    ) -> list[DocumentChunk]:
        async with get_session() as session:
            stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            if chunk_type is not None:
                stmt = stmt.where(DocumentChunk.chunk_type == chunk_type)
            stmt = stmt.order_by(col(DocumentChunk.chunk_index).asc())
            result = await session.exec(stmt)
            return list(result.all())

    async def fetch_pairs_for_job(self, job_id: int) -> list[ComparePair]:
        async with get_session() as session:
            stmt = select(ComparePair).where(ComparePair.job_id == job_id)
            result = await session.exec(stmt)
            return list(result.all())

    async def fetch_match_report(self, pair_id: int) -> dict:
        async with get_session() as session:
            pair = await session.get(ComparePair, pair_id)
            if not pair:
                raise NoResultFound(f"ComparePair {pair_id} not found")

            document_ids = {pair.left_document_id, pair.right_document_id}
            documents = {
                document.id: document
                for document in (await session.exec(select(Document).where(col(Document.id).in_(document_ids)))).all()
            }

            stmt = select(MatchGroup).where(MatchGroup.pair_id == pair_id)
            groups = (await session.exec(stmt)).all()

            group_ids = [group.id for group in groups]
            details: list[MatchDetail] = []
            if group_ids:
                detail_stmt = select(MatchDetail).where(col(MatchDetail.group_id).in_(group_ids))
                details = list((await session.exec(detail_stmt)).all())

            detail_map: dict[int, list] = {}
            for detail in details:
                detail_map.setdefault(detail.group_id, []).append(detail)

            return {
                "pair": pair,
                "left_document": documents.get(pair.left_document_id),
                "right_document": documents.get(pair.right_document_id),
                "groups": [
                    {
                        "group": group,
                        "details": detail_map.get(group.id or 0, []),
                    }
                    for group in groups
                ],
            }

    async def delete_documents(self, document_ids: Iterable[int]) -> None:
        doc_ids = list(document_ids)
        if not doc_ids:
            return

        async with get_session() as session:
            chunk_result = await session.exec(
                select(DocumentChunk.id).where(col(DocumentChunk.document_id).in_(doc_ids))
            )
            chunk_ids = list(chunk_result.all())

            pair_result = await session.exec(
                select(ComparePair.id).where(
                    or_(
                        col(ComparePair.left_document_id).in_(doc_ids),
                        col(ComparePair.right_document_id).in_(doc_ids),
                    )
                )
            )
            pair_ids = list(pair_result.all())

            if pair_ids:
                group_result = await session.exec(
                    select(MatchGroup.id).where(col(MatchGroup.pair_id).in_(pair_ids))
                )
                group_ids = list(group_result.all())
                if group_ids:
                    await session.execute(delete(MatchDetail).where(col(MatchDetail.group_id).in_(group_ids)))
                    await session.execute(delete(MatchGroup).where(col(MatchGroup.id).in_(group_ids)))
                await session.execute(delete(ComparePair).where(col(ComparePair.id).in_(pair_ids)))

            if chunk_ids:
                await session.execute(delete(ChunkEmbedding).where(col(ChunkEmbedding.chunk_id).in_(chunk_ids)))
                await session.execute(delete(DocumentChunk).where(col(DocumentChunk.id).in_(chunk_ids)))

            await session.execute(delete(Document).where(col(Document.id).in_(doc_ids)))
            await session.commit()

    async def delete_pairs(self, pair_ids: Iterable[int]) -> None:
        ids = [int(pid) for pid in pair_ids]
        if not ids:
            return

        async with get_session() as session:
            group_result = await session.exec(
                select(MatchGroup.id).where(col(MatchGroup.pair_id).in_(ids))
            )
            group_ids = list(group_result.all())

            if group_ids:
                await session.execute(delete(MatchDetail).where(col(MatchDetail.group_id).in_(group_ids)))
                await session.execute(delete(MatchGroup).where(col(MatchGroup.id).in_(group_ids)))

            await session.execute(delete(ComparePair).where(col(ComparePair.id).in_(ids)))
            await session.commit()

    async def delete_job(self, job_id: int) -> None:
        async with get_session() as session:
            pair_result = await session.exec(
                select(ComparePair.id).where(ComparePair.job_id == job_id)
            )
            pair_ids = [pid for pid in pair_result.all() if pid is not None]

        if pair_ids:
            await self.delete_pairs(pair_ids)

        async with get_session() as session:
            await session.execute(delete(CompareJob).where(col(CompareJob.id) == job_id))
            await session.commit()

    async def fetch_compare_job(self, job_id: int) -> CompareJob | None:
        async with get_session() as session:
            return await session.get(CompareJob, job_id)

    async def fetch_pair(self, pair_id: int) -> ComparePair | None:
        async with get_session() as session:
            return await session.get(ComparePair, pair_id)
