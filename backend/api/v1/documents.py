"""Document library APIs supporting upload and version retrieval."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status, Response
from pydantic import BaseModel

from backend.core.logging import get_logger
from backend.db.models import Document, DocumentStatus
from backend.services.detection_orchestrator import DetectionOrchestrator

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["Document Library"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _orchestrator() -> DetectionOrchestrator:
    return DetectionOrchestrator()


def _sanitize_filename(filename: Optional[str]) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext or 'unknown'}")
    return name


async def _save_upload(upload: UploadFile, prefix: str) -> str:
    safe_name = _sanitize_filename(upload.filename)
    ext = Path(safe_name).suffix
    tmp = tempfile.NamedTemporaryFile(prefix=f"{prefix}_", suffix=ext, delete=False, dir=tempfile.gettempdir())
    temp_path = Path(tmp.name)
    bytes_written = 0
    try:
        chunk_size = 1024 * 1024
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 20MB limit")
            tmp.write(chunk)
        tmp.flush()
        return str(temp_path)
    except Exception:
        tmp.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        tmp.close()


class DocumentSummary(BaseModel):
    id: int
    project_id: int
    title: Optional[str]
    filename: Optional[str]
    source: Optional[str]
    language: Optional[str]
    status: DocumentStatus
    paragraph_count: int
    sentence_count: int
    char_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, document: Document) -> "DocumentSummary":
        return cls(
            id=document.id,
            project_id=document.project_id,
            title=document.title,
            filename=document.filename,
            source=document.source,
            language=document.language,
            status=document.status,
            paragraph_count=document.paragraph_count,
            sentence_count=document.sentence_count,
            char_count=document.char_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class UploadResponse(BaseModel):
    items: List[DocumentSummary]


@router.post("", response_model=UploadResponse, summary="Upload one or more documents")
async def upload_documents(
    files: List[UploadFile] = File(..., description="Documents to ingest"),
    project_id: int = Form(..., description="Project identifier"),
    source: Optional[str] = Form(None),
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    temp_paths: List[str] = []
    results: List[DocumentSummary] = []
    try:
        for index, upload in enumerate(files):
            temp_path = await _save_upload(upload, f"doc{index}")
            temp_paths.append(temp_path)

            try:
                uploaded = await orchestrator.ingest_file(
                    temp_path,
                    project_id=project_id,
                    title=Path(upload.filename or '').stem,
                    source=source,
                )
                results.append(DocumentSummary.from_model(uploaded.document))
            except Exception as exc:
                logger.error("Document ingestion failed", filename=upload.filename, error=str(exc))
                raise HTTPException(status_code=500, detail=f"Failed to ingest {upload.filename}: {exc}")
    finally:
        for path in temp_paths:
            try:
                if Path(path).exists():
                    os.unlink(path)
            except Exception as cleanup_error:
                logger.warning("Failed to clean temp file", path=path, error=str(cleanup_error))

    return UploadResponse(items=results)


class DocumentListResponse(BaseModel):
    items: List[DocumentSummary]


@router.get("", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    status: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    project_id: Optional[int] = Query(None, description="Filter by project"),
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> DocumentListResponse:
    documents = await orchestrator.list_documents(status=status, project_id=project_id)
    summaries = [DocumentSummary.from_model(document) for document in documents]
    return DocumentListResponse(items=summaries)


class DocumentDetailResponse(BaseModel):
    document: DocumentSummary
    processed_text: Optional[str]
    metadata: Optional[dict]


@router.get("/{document_id}", response_model=DocumentDetailResponse, summary="Get document detail")
async def get_document(
    document_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> DocumentDetailResponse:
    document = await orchestrator.fetch_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    summary = DocumentSummary.from_model(document)
    return DocumentDetailResponse(
        document=summary,
        processed_text=document.processed_text,
        metadata=document.metadata_json,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its versions",
    response_class=Response,
)
async def delete_document(
    document_id: int,
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> Response:
    await orchestrator.delete_documents([document_id])
    return Response(status_code=status.HTTP_204_NO_CONTENT)
