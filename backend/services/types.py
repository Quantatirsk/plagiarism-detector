"""Shared dataclasses used across services."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class SpanPayload:
    """Character span alignment for highlighting."""

    left_start: int
    left_end: int
    right_start: int
    right_end: int
    diff_patch: str | None = None


@dataclass(slots=True)
class CandidatePayload:
    """Rough candidate match data stored ahead of reranking."""

    left_chunk_id: int
    right_chunk_id: int
    rough_method: str
    rough_score: float
    extras: dict | None = None


@dataclass(slots=True)
class EvidencePayload:
    """Detailed scoring result for a candidate match."""

    candidate_id: int
    semantic_score: float | None
    alignment_ratio: float | None
    final_score: float | None
    extra_json: dict | None = None
    spans: Sequence[SpanPayload] | None = None

