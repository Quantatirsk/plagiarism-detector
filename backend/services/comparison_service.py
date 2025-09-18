"""Comparison execution pipeline orchestrating pairwise detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from backend.db.models import ChunkGranularity, ComparePairStatus, DocumentChunk
from backend.services.base_service import BaseService, singleton
from backend.services.detection_orchestrator import DetectionOrchestrator
from backend.services.match_aggregator import MatchAggregator, MatchState
from backend.services.similarity_pipeline import (
    CandidateState,
    PipelineConfig,
    SimilarityPipeline,
)
from backend.services.cross_encoder_service import CrossEncoderService
from backend.services.embedding_service import EmbeddingService
from backend.services.types import CandidatePayload, SpanPayload
from backend.services.text_processor import TextProcessor
from backend.services.storage_gateway import MatchGroupCreate, MatchDetailCreate
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    granularity: ChunkGranularity = ChunkGranularity.PARAGRAPH


@singleton
class ComparisonService(BaseService):
    """Executes pairwise comparison flow leveraging similarity pipeline."""

    def _initialize(self) -> None:
        self.orchestrator = DetectionOrchestrator()
        self.embedding_service = EmbeddingService()
        self.cross_encoder_service = CrossEncoderService()
        self.text_processor = TextProcessor()

    async def run_pair(self, pair_id: int, config: ComparisonConfig) -> None:
        self._ensure_initialized()

        try:
            pair = await self.orchestrator.update_pair_status(pair_id, status=ComparePairStatus.RUNNING)

            left_chunks = await self.orchestrator.fetch_chunks(pair.left_document_id)
            right_chunks = await self.orchestrator.fetch_chunks(pair.right_document_id)

            left_sentences = [chunk for chunk in left_chunks if chunk.chunk_type == ChunkGranularity.SENTENCE]
            right_sentences = [chunk for chunk in right_chunks if chunk.chunk_type == ChunkGranularity.SENTENCE]
            left_paragraphs = [chunk for chunk in left_chunks if chunk.chunk_type == ChunkGranularity.PARAGRAPH]
            right_paragraphs = [chunk for chunk in right_chunks if chunk.chunk_type == ChunkGranularity.PARAGRAPH]

            left_map = {chunk.id: chunk for chunk in left_chunks}
            right_map = {chunk.id: chunk for chunk in right_chunks}

            target_left, target_right = self._select_granularity(
                config.granularity,
                left_sentences,
                right_sentences,
                left_paragraphs,
                right_paragraphs,
            )

            pipeline = SimilarityPipeline(config=config.pipeline)
            candidate_states = await pipeline.run(
                plan_id=pair.id,
                left_chunks=target_left,
                right_chunks=target_right,
                embedding_service=self.embedding_service,
                cross_encoder_service=self.cross_encoder_service,
            )

            aggregator = self._build_aggregator(left_map, right_map)
            threshold = config.pipeline.final_threshold
            for state in candidate_states.values():
                final_score = self._determine_final_score(state)
                if final_score is None or final_score < threshold:
                    continue
                aggregator.add(
                    state.payload.left_chunk_id,
                    state.payload.right_chunk_id,
                    MatchState(
                        final_score=final_score,
                        semantic_score=state.semantic_score,
                        lexical_overlap=state.lexical_overlap,
                        cross_score=state.cross_score,
                        spans=state.spans or self._build_full_spans(state.payload, left_map, right_map),
                    ),
                )

            results = aggregator.finalize()
            group_payloads = [
                self._to_group_payload(pair.id, entry)
                for entry in results
            ]

            detail_payloads = [
                self._to_detail_payload(pair.id, entry, detail)
                for entry in results
                for detail in entry["details"]
            ]

            await self.orchestrator.persist_match_results(pair.id, group_payloads, detail_payloads)

            await self.orchestrator.update_pair_status(
                pair_id,
                status=ComparePairStatus.COMPLETED,
                metrics=self._build_metrics(results),
            )
        except Exception as exc:
            await self.orchestrator.update_pair_status(
                pair_id,
                status=ComparePairStatus.FAILED,
                metrics={"error": str(exc)},
            )
            logger.error("Comparison pair failed", pair_id=pair_id, error=str(exc))
            raise

    def _determine_final_score(self, state: CandidateState) -> Optional[float]:
        """Prefer fused scores but fall back to strongest available signal."""
        for score in (
            state.final_score,
            state.cross_score,
            state.semantic_score,
            state.lexical_overlap,
        ):
            if score is not None:
                return float(score)
        try:
            return float(state.payload.rough_score)
        except (AttributeError, TypeError, ValueError):
            return None

    def _build_aggregator(
        self,
        left_map: Dict[int, DocumentChunk],
        right_map: Dict[int, DocumentChunk],
    ) -> MatchAggregator:
        left_lookup = self._build_parent_lookup(left_map)
        right_lookup = self._build_parent_lookup(right_map)
        return MatchAggregator(
            left_map=left_map,
            right_map=right_map,
            left_lookup=left_lookup,
            right_lookup=right_lookup,
        )

    def _build_parent_lookup(self, chunks: Dict[int, DocumentChunk]) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        paragraphs = [chunk for chunk in chunks.values() if chunk.chunk_type == ChunkGranularity.PARAGRAPH]
        sentences = [chunk for chunk in chunks.values() if chunk.chunk_type == ChunkGranularity.SENTENCE]
        paragraph_map = {paragraph.id: paragraph for paragraph in paragraphs}
        paragraphs_sorted = sorted(paragraphs, key=lambda chunk: chunk.start_pos)

        for paragraph in paragraphs_sorted:
            lookup[paragraph.id] = paragraph.id

        for sentence in sentences:
            parent_id = sentence.parent_chunk_id
            if parent_id is not None and parent_id in paragraph_map:
                lookup[sentence.id] = parent_id
                continue
            fallback = self._locate_parent_by_offsets(sentence, paragraphs_sorted)
            lookup[sentence.id] = fallback if fallback is not None else sentence.id
        return lookup

    def _locate_parent_by_offsets(
        self,
        sentence: DocumentChunk,
        paragraphs: Sequence[DocumentChunk],
    ) -> Optional[int]:
        for paragraph in paragraphs:
            if paragraph.start_pos <= sentence.start_pos and paragraph.end_pos >= sentence.end_pos:
                return paragraph.id
        if not paragraphs:
            return None
        closest = min(paragraphs, key=lambda chunk: abs(chunk.start_pos - sentence.start_pos))
        return closest.id

    def _build_full_spans(
        self,
        payload: CandidatePayload,
        left_map: Dict[int, DocumentChunk],
        right_map: Dict[int, DocumentChunk],
    ) -> List[SpanPayload]:
        left_chunk = left_map[payload.left_chunk_id]
        right_chunk = right_map[payload.right_chunk_id]
        return [
            SpanPayload(
                left_start=0,
                left_end=len(left_chunk.text),
                right_start=0,
                right_end=len(right_chunk.text),
            )
        ]

    def _to_group_payload(self, pair_id: int, entry: Dict[str, object]):
        return MatchGroupCreate(
            pair_id=pair_id,
            left_chunk_id=entry["left_chunk_id"],
            right_chunk_id=entry["right_chunk_id"],
            final_score=entry.get("final_score"),
            semantic_score=entry.get("semantic_score"),
            lexical_overlap=entry.get("lexical_overlap"),
            cross_score=entry.get("cross_score"),
            alignment_ratio=entry.get("alignment_ratio"),
            span_count=entry.get("span_count", 0),
            match_count=entry.get("match_count", 0),
            paragraph_spans=entry.get("paragraph_spans"),
            document_spans=entry.get("document_spans"),
        )

    def _to_detail_payload(self, pair_id: int, entry: Dict[str, object], detail: Dict[str, object]):
        return MatchDetailCreate(
            left_chunk_id=detail["left_chunk_id"],
            right_chunk_id=detail["right_chunk_id"],
            final_score=detail.get("final_score"),
            semantic_score=detail.get("semantic_score"),
            lexical_overlap=detail.get("lexical_overlap"),
            cross_score=detail.get("cross_score"),
            spans=detail.get("spans"),
            group_key=detail.get("group_pair") or (entry["left_chunk_id"], entry["right_chunk_id"]),
        )

    def _build_metrics(self, groups: Sequence[Dict[str, object]]) -> Dict[str, object]:
        total = len(groups)
        top_score = max((group.get("final_score") or 0.0 for group in groups), default=0.0)
        coverage = sum(group.get("alignment_ratio") or 0.0 for group in groups)
        return {
            "group_count": total,
            "top_score": round(top_score, 4),
            "aggregate_alignment": round(coverage, 4),
        }

    def _select_granularity(
        self,
        granularity: ChunkGranularity,
        left_sentences: List[DocumentChunk],
        right_sentences: List[DocumentChunk],
        left_paragraphs: List[DocumentChunk],
        right_paragraphs: List[DocumentChunk],
    ) -> tuple[List[DocumentChunk], List[DocumentChunk]]:
        if granularity == ChunkGranularity.PARAGRAPH:
            if left_paragraphs and right_paragraphs:
                return left_paragraphs, right_paragraphs
            return (
                left_paragraphs or left_sentences,
                right_paragraphs or right_sentences,
            )
        if left_sentences and right_sentences:
            return left_sentences, right_sentences
        return (
            left_sentences or left_paragraphs,
            right_sentences or right_paragraphs,
        )
