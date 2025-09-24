"""Database models for document library and pairwise comparison workflow."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Column, Enum as SAEnum, JSON, Text
from sqlmodel import Field, Relationship, SQLModel


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DocumentStatus(str, Enum):
    """Lifecycle state for an ingested document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkGranularity(str, Enum):
    """Granularity of stored text segments."""

    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class CompareJobStatus(str, Enum):
    """Processing stage for a comparison job."""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparePairStatus(str, Enum):
    """Processing state for an individual comparison pair."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class Project(SQLModel, table=True):
    """Grouping for related documents and comparison jobs."""

    __tablename__ = "project"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1024)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    documents: List["Document"] = Relationship(back_populates="project")
    jobs: List["CompareJob"] = Relationship(back_populates="project")


# ---------------------------------------------------------------------------
# Document library
# ---------------------------------------------------------------------------


class Document(SQLModel, table=True):
    """Single uploaded document with processing metadata."""

    __tablename__ = "document"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", nullable=False, index=True)
    title: Optional[str] = Field(default=None, max_length=255)
    filename: Optional[str] = Field(default=None, max_length=255)
    source: Optional[str] = Field(default=None, max_length=120)
    checksum: str = Field(max_length=128, index=True)
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        sa_column=Column(SAEnum(DocumentStatus, name="document_status"), nullable=False),
    )
    language: Optional[str] = Field(default=None, max_length=16)
    char_count: int = Field(default=0, nullable=False)
    paragraph_count: int = Field(default=0, nullable=False)
    sentence_count: int = Field(default=0, nullable=False)
    processed_text: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    storage_path: Optional[str] = Field(default=None, max_length=500)
    metadata_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    completed_at: Optional[datetime] = Field(default=None)

    project: Project = Relationship(back_populates="documents")
    chunks: List["DocumentChunk"] = Relationship(back_populates="document")
    left_pairs: List["ComparePair"] = Relationship(
        back_populates="left_document", sa_relationship_kwargs={"foreign_keys": "ComparePair.left_document_id"}
    )
    right_pairs: List["ComparePair"] = Relationship(
        back_populates="right_document", sa_relationship_kwargs={"foreign_keys": "ComparePair.right_document_id"}
    )


class DocumentChunk(SQLModel, table=True):
    """Paragraph or sentence chunk with offsets for a document version."""

    __tablename__ = "document_chunk"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", nullable=False, index=True)
    chunk_type: ChunkGranularity = Field(
        sa_column=Column(SAEnum(ChunkGranularity, name="chunk_granularity"), nullable=False)
    )
    chunk_index: int = Field(default=0, nullable=False)
    start_pos: int = Field(default=0, nullable=False)
    end_pos: int = Field(default=0, nullable=False)
    parent_chunk_id: Optional[int] = Field(default=None, foreign_key="document_chunk.id", index=True)
    text: str = Field(sa_column=Column(Text, nullable=False))
    text_hash: Optional[str] = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    document: Document = Relationship(back_populates="chunks")
    parent_chunk: Optional["DocumentChunk"] = Relationship(
        sa_relationship_kwargs={"remote_side": "DocumentChunk.id"}
    )
    embeddings: List["ChunkEmbedding"] = Relationship(back_populates="chunk")


class ChunkEmbedding(SQLModel, table=True):
    """Embedding metadata stored alongside vector index ids."""

    __tablename__ = "chunk_embedding"

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="document_chunk.id", nullable=False, unique=True)
    vector_id: str = Field(max_length=64, nullable=False, index=True)
    model: str = Field(max_length=120, nullable=False)
    dimension: int = Field(nullable=False)
    norm: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    chunk: DocumentChunk = Relationship(back_populates="embeddings")


# ---------------------------------------------------------------------------
# Comparison workflow
# ---------------------------------------------------------------------------


class CompareJob(SQLModel, table=True):
    """Logical collection of many pairwise comparisons."""

    __tablename__ = "compare_job"

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", nullable=False, index=True)
    name: Optional[str] = Field(default=None, max_length=200)
    status: CompareJobStatus = Field(
        default=CompareJobStatus.DRAFT,
        sa_column=Column(SAEnum(CompareJobStatus, name="compare_job_status"), nullable=False),
    )
    config_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    project: Project = Relationship(back_populates="jobs")
    pairs: List["ComparePair"] = Relationship(back_populates="job")


class ComparePair(SQLModel, table=True):
    """Represents the comparison of two documents."""

    __tablename__ = "compare_pair"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="compare_job.id", nullable=False, index=True)
    left_document_id: int = Field(foreign_key="document.id", nullable=False, index=True)
    right_document_id: int = Field(foreign_key="document.id", nullable=False, index=True)
    status: ComparePairStatus = Field(
        default=ComparePairStatus.PENDING,
        sa_column=Column(SAEnum(ComparePairStatus, name="compare_pair_status"), nullable=False),
    )
    metrics_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    job: CompareJob = Relationship(back_populates="pairs")
    left_document: Document = Relationship(
        back_populates="left_pairs", sa_relationship_kwargs={"foreign_keys": "ComparePair.left_document_id"}
    )
    right_document: Document = Relationship(
        back_populates="right_pairs", sa_relationship_kwargs={"foreign_keys": "ComparePair.right_document_id"}
    )
    match_groups: List["MatchGroup"] = Relationship(back_populates="pair")


class MatchGroup(SQLModel, table=True):
    """Paragraph-level match aggregation between two document versions."""

    __tablename__ = "match_group"

    id: Optional[int] = Field(default=None, primary_key=True)
    pair_id: int = Field(foreign_key="compare_pair.id", nullable=False, index=True)
    left_chunk_id: int = Field(foreign_key="document_chunk.id", nullable=False, index=True)
    right_chunk_id: int = Field(foreign_key="document_chunk.id", nullable=False, index=True)
    final_score: Optional[float] = Field(default=None)
    semantic_score: Optional[float] = Field(default=None)
    cross_score: Optional[float] = Field(default=None)
    alignment_ratio: Optional[float] = Field(default=None)
    span_count: int = Field(default=0, nullable=False)
    match_count: int = Field(default=0, nullable=False)
    paragraph_spans_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    document_spans_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))

    pair: ComparePair = Relationship(back_populates="match_groups")
    details: list["MatchDetail"] = Relationship(back_populates="group")


class MatchDetail(SQLModel, table=True):
    """Sentence-level match detail within a paragraph grouping."""

    __tablename__ = "match_detail"

    id: Optional[int] = Field(default=None, primary_key=True)
    group_id: int = Field(foreign_key="match_group.id", nullable=False, index=True)
    left_chunk_id: int = Field(foreign_key="document_chunk.id", nullable=False, index=True)
    right_chunk_id: int = Field(foreign_key="document_chunk.id", nullable=False, index=True)
    final_score: Optional[float] = Field(default=None)
    semantic_score: Optional[float] = Field(default=None)
    cross_score: Optional[float] = Field(default=None)
    spans_json: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))

    group: MatchGroup = Relationship(back_populates="details")


__all__ = [
    "Project",
    "Document",
    "DocumentChunk",
    "ChunkEmbedding",
    "CompareJob",
    "ComparePair",
    "MatchGroup",
    "MatchDetail",
    "DocumentStatus",
    "ChunkGranularity",
    "CompareJobStatus",
    "ComparePairStatus",
]
