"""Composable similarity stages for multi-document detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.db.models import DocumentChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.cross_encoder_service import CrossEncoderService
from backend.services.types import CandidatePayload, SpanPayload


@dataclass
class PipelineConfig:
    lexical_shingle_size: int = 8
    lexical_threshold: float = 0.4
    semantic_threshold: float = 0.70
    final_threshold: float = 0.75
    top_k: int = 5
    max_candidates: int = 500
    cross_encoder_top_k: int = 200
    cross_encoder_threshold: float = 0.55


@dataclass
class CandidateState:
    payload: CandidatePayload
    lexical_overlap: Optional[float] = None
    semantic_score: Optional[float] = None
    cross_score: Optional[float] = None
    final_score: Optional[float] = None
    spans: List[SpanPayload] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PipelineContext:
    plan_id: int
    left_chunks: Dict[int, DocumentChunk]
    right_chunks: Dict[int, DocumentChunk]
    config: PipelineConfig
    embedding_service: EmbeddingService
    left_embeddings: Dict[int, Sequence[float]]
    right_embeddings: Dict[int, Sequence[float]]
    cross_encoder_service: Optional[CrossEncoderService]
    candidate_states: Dict[Tuple[int, int], CandidateState] = field(default_factory=dict)


class PipelineStage:
    async def run(self, context: PipelineContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class FingerprintRecallStage(PipelineStage):
    """Generate coarse candidates using shingling + Jaccard."""

    async def run(self, context: PipelineContext) -> None:
        shingle_size = context.config.lexical_shingle_size
        left_fingerprints = {
            cid: self._fingerprint(chunk.text, shingle_size)
            for cid, chunk in context.left_chunks.items()
        }
        right_fingerprints = {
            cid: self._fingerprint(chunk.text, shingle_size)
            for cid, chunk in context.right_chunks.items()
        }

        for left_id, left_fp in left_fingerprints.items():
            if not left_fp:
                continue
            for right_id, right_fp in right_fingerprints.items():
                if not right_fp:
                    continue
                score = self._jaccard(left_fp, right_fp)
                if score >= context.config.lexical_threshold:
                    key = (left_id, right_id)
                    payload = CandidatePayload(
                        left_chunk_id=left_id,
                        right_chunk_id=right_id,
                        rough_method="fingerprint",
                        rough_score=float(score),
                        extras={"fingerprint_score": float(score)},
                    )
                    state = context.candidate_states.get(key)
                    if not state:
                        context.candidate_states[key] = CandidateState(
                            payload=payload,
                            lexical_overlap=float(score),
                            metadata={"fingerprint_score": float(score)},
                        )
                    else:
                        state.lexical_overlap = max(state.lexical_overlap or 0.0, float(score))
                        state.metadata["fingerprint_score"] = float(score)
                        state.payload.extras = state.payload.extras or {}
                        state.payload.extras["fingerprint_score"] = float(score)

    def _fingerprint(self, text: str, shingle_size: int) -> set[str]:
        normalized = text.lower().replace("\n", " ")
        tokens = normalized.split()
        if len(tokens) < shingle_size:
            return set(tokens)
        shingles = {
            " ".join(tokens[i : i + shingle_size])
            for i in range(0, len(tokens) - shingle_size + 1)
        }
        return shingles

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        intersection = len(a & b)
        if not intersection:
            return 0.0
        union = len(a | b)
        return intersection / union if union else 0.0


class SemanticRecallStage(PipelineStage):
    """Compute embedding similarities to enrich candidates."""

    async def run(self, context: PipelineContext) -> None:
        embedding_service = context.embedding_service
        left_ids = list(context.left_chunks.keys())
        right_ids = list(context.right_chunks.keys())

        left_vectors = await self._ensure_embeddings(left_ids, context.left_chunks, context.left_embeddings, embedding_service)
        right_vectors = await self._ensure_embeddings(right_ids, context.right_chunks, context.right_embeddings, embedding_service)

        E1 = np.array(left_vectors, dtype=float)
        E2 = np.array(right_vectors, dtype=float)
        if not E1.size or not E2.size:
            return

        E1n = self._normalize(E1)
        E2n = self._normalize(E2)
        similarity = E1n @ E2n.T

        top_k = context.config.top_k
        threshold = context.config.semantic_threshold

        for i, left_id in enumerate(left_ids):
            scores = similarity[i]
            indices = np.argsort(scores)[::-1][:top_k]
            for j in indices:
                right_id = right_ids[j]
                score = float(scores[j])
                if score < threshold:
                    continue
                key = (left_id, right_id)
                payload = CandidatePayload(
                    left_chunk_id=left_id,
                    right_chunk_id=right_id,
                    rough_method="semantic",
                    rough_score=score,
                    extras={"semantic_score": score},
                )
                state = context.candidate_states.get(key)
                if not state:
                    context.candidate_states[key] = CandidateState(
                        payload=payload,
                        semantic_score=score,
                        metadata={"semantic_score": score},
                    )
                else:
                    state.payload.rough_method = "semantic"
                    state.payload.rough_score = score
                    state.payload.extras = state.payload.extras or {}
                    state.payload.extras["semantic_score"] = score
                    state.semantic_score = max(state.semantic_score or 0.0, score)
                    state.metadata["semantic_score"] = score

    async def _ensure_embeddings(
        self,
        ids: List[int],
        chunks: Dict[int, DocumentChunk],
        cache: Dict[int, Sequence[float]],
        embedding_service: EmbeddingService,
    ) -> List[Sequence[float]]:
        missing_ids = [cid for cid in ids if cid not in cache]
        if missing_ids:
            texts = [chunks[cid].text for cid in missing_ids]
            embeddings = await embedding_service.embed_batch(texts)
            for cid, embedding in zip(missing_ids, embeddings):
                cache[cid] = embedding
        return [cache[cid] for cid in ids]

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class CrossEncoderStage(PipelineStage):
    """Apply cross-encoder style reranking to promising candidates."""

    async def run(self, context: PipelineContext) -> None:
        service = context.cross_encoder_service
        if service is None or not context.candidate_states:
            return

        ranked_items = sorted(
            context.candidate_states.items(),
            key=lambda item: self._base_score(item[1]),
            reverse=True,
        )
        top_k = context.config.cross_encoder_top_k
        threshold = context.config.cross_encoder_threshold
        selected = ranked_items[:top_k]
        pairs = [
            (
                context.left_chunks[left_id].text,
                context.right_chunks[right_id].text,
            )
            for (left_id, right_id), _ in selected
        ]
        scores = await service.score_pairs(pairs)

        for ((key, state), score) in zip(selected, scores):
            state.cross_score = score
            state.metadata["cross_score"] = score
            state.payload.extras = state.payload.extras or {}
            state.payload.extras["cross_score"] = score
            if score < threshold:
                state.metadata["cross_encoder_flagged"] = 1.0

    def _base_score(self, state: CandidateState) -> float:
        if state.semantic_score is not None:
            return state.semantic_score
        if state.lexical_overlap is not None:
            return state.lexical_overlap
        return state.payload.rough_score


class FusionScoringStage(PipelineStage):
    """Combine lexical and semantic signals to produce final score."""

    async def run(self, context: PipelineContext) -> None:
        for state in context.candidate_states.values():
            lexical = state.lexical_overlap or 0.0
            semantic = state.semantic_score or 0.0
            if state.cross_score is not None:
                cross = state.cross_score
                final = 0.1 * lexical + 0.6 * semantic + 0.3 * cross
            else:
                final = 0.25 * lexical + 0.75 * semantic
            fused_score = max(final, semantic)
            state.final_score = fused_score
            state.metadata["final_score"] = fused_score
            state.payload.extras = state.payload.extras or {}
            state.payload.extras["final_score"] = fused_score


class AlignmentStage(PipelineStage):
    """Derive span alignments for high scoring matches."""

    MAX_DIFF_CHARS = 4000

    async def run(self, context: PipelineContext) -> None:
        threshold = context.config.final_threshold
        for key, state in context.candidate_states.items():
            if (state.final_score or 0.0) < threshold:
                continue
            left_chunk = context.left_chunks[key[0]]
            right_chunk = context.right_chunks[key[1]]
            spans = self._align(left_chunk.text, right_chunk.text)
            state.spans = spans
            patch = self._diff_patch(left_chunk.text, right_chunk.text)
            if patch:
                trimmed = patch
                if len(trimmed) > self.MAX_DIFF_CHARS:
                    trimmed = f"{trimmed[: self.MAX_DIFF_CHARS]}\n...diff truncated..."
                state.metadata["diff_patch"] = trimmed
                state.payload.extras = state.payload.extras or {}
                state.payload.extras["diff_patch"] = trimmed

    def _align(self, left: str, right: str) -> List[SpanPayload]:
        from difflib import SequenceMatcher

        matcher = SequenceMatcher(None, left, right)
        spans: List[SpanPayload] = []
        for match in matcher.get_matching_blocks():
            if match.size < 3:
                continue
            spans.append(
                SpanPayload(
                    left_start=match.a,
                    left_end=match.a + match.size,
                    right_start=match.b,
                    right_end=match.b + match.size,
                )
            )
        return spans

    def _diff_patch(self, left: str, right: str) -> str:
        import difflib

        left_lines = left.splitlines()
        right_lines = right.splitlines()
        diff = list(
            difflib.unified_diff(
                left_lines,
                right_lines,
                fromfile="left",
                tofile="right",
                lineterm="",
            )
        )
        # First two header lines are always present; skip returning empty string when bodies match.
        if len(diff) <= 2:
            return ""
        return "\n".join(diff)


class SimilarityPipeline:
    """High-level orchestrator for running similarity stages."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stages: List[PipelineStage] = [
            FingerprintRecallStage(),
            SemanticRecallStage(),
            CrossEncoderStage(),
            FusionScoringStage(),
            AlignmentStage(),
        ]

    async def run(
        self,
        plan_id: int,
        left_chunks: Sequence[DocumentChunk],
        right_chunks: Sequence[DocumentChunk],
        embedding_service: EmbeddingService,
        left_embeddings: Optional[Dict[int, Sequence[float]]] = None,
        right_embeddings: Optional[Dict[int, Sequence[float]]] = None,
        cross_encoder_service: Optional[CrossEncoderService] = None,
    ) -> Dict[Tuple[int, int], CandidateState]:
        context = PipelineContext(
            plan_id=plan_id,
            left_chunks={chunk.id: chunk for chunk in left_chunks},
            right_chunks={chunk.id: chunk for chunk in right_chunks},
            config=self.config,
            embedding_service=embedding_service,
            left_embeddings=left_embeddings or {},
            right_embeddings=right_embeddings or {},
            cross_encoder_service=cross_encoder_service,
        )
        for stage in self.stages:
            await stage.run(context)
            self._prune_candidates(context)
        return context.candidate_states

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_candidates(self, context: PipelineContext) -> None:
        """Ensure candidate volume remains bounded without aborting later stages."""
        max_candidates = context.config.max_candidates
        if not max_candidates:
            return
        if len(context.candidate_states) <= max_candidates:
            return

        def score(state: CandidateState) -> float:
            if state.final_score is not None:
                return state.final_score
            if state.cross_score is not None:
                return state.cross_score
            if state.semantic_score is not None:
                return state.semantic_score
            if state.lexical_overlap is not None:
                return state.lexical_overlap
            return state.payload.rough_score

        sorted_items = sorted(
            context.candidate_states.items(),
            key=lambda item: score(item[1]),
            reverse=True,
        )
        context.candidate_states = dict(sorted_items[:max_candidates])
