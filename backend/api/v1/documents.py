"""Document library APIs supporting upload and version retrieval."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status, Response
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


class UploadTaskResponse(BaseModel):
    task_id: str
    documents: List[DocumentSummary]
    message: str = "Documents uploaded and processing started"


@router.post("", response_model=UploadTaskResponse, summary="Upload one or more documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Documents to ingest"),
    project_id: int = Form(..., description="Project identifier"),
    source: Optional[str] = Form(None),
    orchestrator: DetectionOrchestrator = Depends(_orchestrator),
) -> UploadTaskResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    project = await orchestrator.fetch_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Create a parent progress task for batch upload
    from backend.services.progress_tracker import ProgressTracker, ProgressType
    progress_tracker = ProgressTracker()

    batch_task_id = progress_tracker.create_task(
        task_type=ProgressType.BATCH_PROCESSING,
        description=f"正在上传 {len(files)} 个文档到项目 {project_id}",
        total_steps=len(files),
        metadata={"project_id": project_id, "file_count": len(files)}
    )

    # Save uploaded files first and create placeholder documents
    temp_paths: List[str] = []
    placeholder_docs: List[DocumentSummary] = []

    try:
        for index, upload in enumerate(files):
            temp_path = await _save_upload(upload, f"doc{index}")
            temp_paths.append(temp_path)

            # Create placeholder document
            from backend.db.models import Document
            from backend.db import get_session
            import hashlib

            # Generate a temporary checksum for the placeholder
            temp_checksum = hashlib.sha256(f"{upload.filename}_{index}_{project_id}".encode()).hexdigest()

            async with get_session() as session:
                doc = Document(
                    project_id=project_id,
                    title=Path(upload.filename or '').stem,
                    filename=upload.filename,
                    source=source,
                    checksum=temp_checksum,  # Will be updated after processing
                    status=DocumentStatus.PENDING,
                    paragraph_count=0,
                    sentence_count=0,
                    char_count=0,
                )
                session.add(doc)
                await session.commit()
                await session.refresh(doc)
                placeholder_docs.append(DocumentSummary.from_model(doc))

        # Process documents asynchronously
        async def process_documents():
            await progress_tracker.start_task(batch_task_id)
            results = []

            for index, (temp_path, placeholder) in enumerate(zip(temp_paths, placeholder_docs)):
                try:
                    # Update parent task progress
                    await progress_tracker.update_progress(
                        batch_task_id,
                        current_step=index,
                        message=f"正在处理文档 {index + 1}/{len(files)}: {placeholder.filename}"
                    )

                    # Update placeholder to processing status
                    await orchestrator.mark_document_status(placeholder.id, status=DocumentStatus.PROCESSING)

                    # Process the document for existing placeholder
                    uploaded = await orchestrator.process_existing_document(
                        temp_path,
                        document_id=placeholder.id,
                        project_id=project_id,
                    )
                    results.append(uploaded)

                except Exception as exc:
                    logger.error("Document ingestion failed",
                               document_id=placeholder.id,
                               filename=placeholder.filename,
                               error=str(exc))
                    # Mark as failed
                    await orchestrator.mark_document_status(placeholder.id, status=DocumentStatus.FAILED)
                finally:
                    # Cleanup temp file
                    try:
                        if Path(temp_path).exists():
                            os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.warning("Failed to clean temp file", path=temp_path, error=str(cleanup_error))

            # Complete the batch task
            await progress_tracker.complete_task(batch_task_id, message=f"成功处理 {len(results)} 个文档")
            return results

        background_tasks.add_task(process_documents)

    except Exception as e:
        # Cleanup on immediate failure
        for path in temp_paths:
            try:
                if Path(path).exists():
                    os.unlink(path)
            except Exception:
                pass
        raise

    return UploadTaskResponse(
        task_id=batch_task_id,
        documents=placeholder_docs,
        message=f"Uploaded {len(files)} document(s), processing in background"
    )


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
