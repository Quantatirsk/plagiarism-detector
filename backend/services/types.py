"""Shared dataclasses used across services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(slots=True)
class SpanPayload:
    """Character span alignment for highlighting."""

    left_start: int
    left_end: int
    right_start: int
    right_end: int
    diff_patch: Optional[str] = None


@dataclass(slots=True)
class CandidatePayload:
    """Rough candidate match data stored ahead of reranking."""

    left_chunk_id: int
    right_chunk_id: int
    rough_method: str
    rough_score: float
    extras: Optional[dict] = None


@dataclass(slots=True)
class EvidencePayload:
    """Detailed scoring result for a candidate match."""

    candidate_id: int
    semantic_score: Optional[float]
    alignment_ratio: Optional[float]
    final_score: Optional[float]
    extra_json: Optional[dict] = None
    spans: Optional[Sequence[SpanPayload]] = None

