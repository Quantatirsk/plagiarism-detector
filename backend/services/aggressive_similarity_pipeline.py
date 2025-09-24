"""Aggressive similarity pipeline with simplified 3-stage processing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.db.models import DocumentChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.cross_encoder_service import CrossEncoderService
from backend.services.types import CandidatePayload, SpanPayload
from backend.services.minhash_filter import MinHashFilterStage as MinHashService
from backend.services.pipeline_metrics import metrics_collector


@dataclass
class AggressivePipelineConfig:
    """Simplified configuration with fewer parameters."""
    semantic_threshold: float = 0.65  # Lower threshold for better recall
    final_threshold: float = 0.75
    top_k: int = 50  # Increased from 5 to 50 for better recall
    # MinHash parameters (optional stage)
    use_minhash: bool = True  # Enabled by default for better performance
    minhash_threshold: float = 0.3
    # Cross-encoder parameters
    cross_encoder_threshold: float = 0.55


@dataclass
class CandidateState:
    """Simplified candidate state tracking."""
    payload: CandidatePayload
    semantic_score: Optional[float] = None
    minhash_score: Optional[float] = None  # Optional MinHash score
    cross_score: Optional[float] = None
    final_score: Optional[float] = None
    spans: List[SpanPayload] = field(default_factory=list)


@dataclass
class AggressivePipelineContext:
    """Simplified pipeline context."""
    plan_id: int
    left_chunks: Dict[int, DocumentChunk]
    right_chunks: Dict[int, DocumentChunk]
    config: AggressivePipelineConfig
    embedding_service: EmbeddingService
    left_embeddings: Dict[int, Sequence[float]]
    right_embeddings: Dict[int, Sequence[float]]
    cross_encoder_service: Optional[CrossEncoderService]
    candidate_states: Dict[Tuple[int, int], CandidateState] = field(default_factory=dict)


class PipelineStage:
    """Base class for pipeline stages."""
    async def run(self, context: AggressivePipelineContext) -> None:
        raise NotImplementedError


class SemanticRecallStage(PipelineStage):
    """Stage 1: Semantic similarity with aggressive recall settings."""

    async def run(self, context: AggressivePipelineContext) -> None:
        embedding_service = context.embedding_service
        left_ids = list(context.left_chunks.keys())
        right_ids = list(context.right_chunks.keys())

        # Get embeddings
        left_vectors = await self._ensure_embeddings(
            left_ids, context.left_chunks, context.left_embeddings, embedding_service
        )
        right_vectors = await self._ensure_embeddings(
            right_ids, context.right_chunks, context.right_embeddings, embedding_service
        )

        E1 = np.array(left_vectors, dtype=float)
        E2 = np.array(right_vectors, dtype=float)
        if not E1.size or not E2.size:
            return

        # Normalize vectors
        E1n = self._normalize(E1)
        E2n = self._normalize(E2)
        similarity = E1n @ E2n.T

        # Aggressive recall: lower threshold, higher top_k
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
                context.candidate_states[key] = CandidateState(
                    payload=payload,
                    semantic_score=score,
                )

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


class MinHashFilterStage(PipelineStage):
    """Stage 2 (Optional): MinHash-based filtering for efficiency."""

    def __init__(self):
        self.service = MinHashService()

    async def run(self, context: AggressivePipelineContext) -> None:
        if not context.config.use_minhash:
            return

        # Prepare text data for MinHash
        left_texts = {cid: chunk.text for cid, chunk in context.left_chunks.items()}
        right_texts = {cid: chunk.text for cid, chunk in context.right_chunks.items()}

        # Process signatures
        left_signatures = await self.service.process_chunks(left_texts)
        right_signatures = await self.service.process_chunks(right_texts)

        # Find similar pairs
        similar_pairs = await self.service.find_similar_pairs(
            left_signatures,
            right_signatures,
            threshold=context.config.minhash_threshold
        )

        # Update candidate states with MinHash scores
        for left_id, right_id, minhash_score in similar_pairs:
            key = (left_id, right_id)
            if key in context.candidate_states:
                # Enhance existing candidate
                context.candidate_states[key].minhash_score = minhash_score
            else:
                # Create new candidate from MinHash
                payload = CandidatePayload(
                    left_chunk_id=left_id,
                    right_chunk_id=right_id,
                    rough_method="minhash",
                    rough_score=minhash_score,
                    extras={"minhash_score": minhash_score},
                )
                context.candidate_states[key] = CandidateState(
                    payload=payload,
                    minhash_score=minhash_score,
                )

        # Optional: Filter out low-scoring semantic candidates if MinHash disagrees
        if context.config.minhash_threshold > 0:
            filtered_states = {}
            for key, state in context.candidate_states.items():
                # Keep if has good semantic score OR good MinHash score
                semantic_ok = (state.semantic_score or 0) >= context.config.semantic_threshold
                minhash_ok = (state.minhash_score or 0) >= context.config.minhash_threshold
                if semantic_ok or minhash_ok:
                    filtered_states[key] = state
            context.candidate_states = filtered_states


class CrossEncoderDirectStage(PipelineStage):
    """Stage 3: Mandatory cross-encoder scoring with complete trust."""

    async def run(self, context: AggressivePipelineContext) -> None:
        if not context.candidate_states:
            return

        service = context.cross_encoder_service
        if service is None:
            # This should not happen since cross-encoder is mandatory
            raise RuntimeError("Cross-encoder service is required but not available")

        # Score all candidates with cross-encoder
        pairs = [
            (
                context.left_chunks[left_id].text,
                context.right_chunks[right_id].text,
            )
            for (left_id, right_id) in context.candidate_states.keys()
        ]

        scores = await service.score_pairs(pairs)

        # Update scores with cross-encoder results
        for (key, score) in zip(context.candidate_states.keys(), scores):
            state = context.candidate_states[key]
            state.cross_score = score

            # Use cross-encoder score as the final score
            state.final_score = score


class AlignmentStage(PipelineStage):
    """Simplified alignment stage that generates paragraph-level spans only."""

    def __init__(self):
        # Simplified to always use full paragraph matching
        pass

    async def run(self, context: AggressivePipelineContext) -> None:
        threshold = context.config.final_threshold
        for key, state in context.candidate_states.items():
            if (state.final_score or 0.0) < threshold:
                continue

            left_chunk = context.left_chunks[key[0]]
            right_chunk = context.right_chunks[key[1]]

            # Always use full paragraph matching instead of fine-grained alignment
            # This simplifies the matching to paragraph-level only
            state.spans = [
                SpanPayload(
                    left_start=0,
                    left_end=len(left_chunk.text),
                    right_start=0,
                    right_end=len(right_chunk.text),
                )
            ]


class AggressiveSimilarityPipeline:
    """Simplified 3-stage pipeline: Semantic → (MinHash) → Cross-encoder."""

    def __init__(self, config: Optional[AggressivePipelineConfig] = None):
        self.config = config or AggressivePipelineConfig()
        self.stages: List[PipelineStage] = [
            SemanticRecallStage(),
            MinHashFilterStage(),
            CrossEncoderDirectStage(),
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
        # Start metrics collection
        pipeline_id = f"aggressive-{plan_id}"
        metrics_collector.start_pipeline(pipeline_id)

        context = AggressivePipelineContext(
            plan_id=plan_id,
            left_chunks={chunk.id: chunk for chunk in left_chunks},
            right_chunks={chunk.id: chunk for chunk in right_chunks},
            config=self.config,
            embedding_service=embedding_service,
            left_embeddings=left_embeddings or {},
            right_embeddings=right_embeddings or {},
            cross_encoder_service=cross_encoder_service,
        )

        # Track initial candidates (all possible pairs)
        initial_candidates = len(left_chunks) * len(right_chunks)
        metrics_collector.current_metrics.total_candidates_initial = initial_candidates

        for stage in self.stages:
            stage_name = stage.__class__.__name__
            candidates_before = len(context.candidate_states)

            # Start stage metrics
            metrics_collector.start_stage(stage_name, candidates_before)

            # Run stage
            await stage.run(context)

            # End stage metrics
            candidates_after = len(context.candidate_states)

            # Get stage-specific metrics
            api_calls = 0
            cache_hits = 0
            cache_misses = 0

            if isinstance(stage, CrossEncoderDirectStage) and cross_encoder_service:
                stats = cross_encoder_service.get_cache_stats()
                cache_hits = stats.get("cache_hits", 0)
                cache_misses = stats.get("cache_misses", 0)
                api_calls = cache_misses  # Each miss results in an API call

            metrics_collector.end_stage(
                stage_name,
                candidates_after,
                api_calls=api_calls,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )

        # Filter by final threshold
        filtered = {
            key: state
            for key, state in context.candidate_states.items()
            if (state.final_score or 0.0) >= context.config.final_threshold
        }

        # End pipeline metrics
        metrics = metrics_collector.end_pipeline(len(filtered))

        return filtered