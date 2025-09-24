"""Comparison execution pipeline orchestrating pairwise detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from backend.db.models import ChunkGranularity, ComparePairStatus, DocumentChunk
from backend.services.base_service import BaseService, singleton
from backend.services.detection_orchestrator import DetectionOrchestrator
from backend.services.match_aggregator import MatchAggregator, MatchState
# Legacy pipeline imports removed
from backend.services.aggressive_similarity_pipeline import (
    AggressiveSimilarityPipeline,
    AggressivePipelineConfig,
    CandidateState as AggressiveCandidateState,
)
from backend.services.cross_encoder_service import CrossEncoderService
from backend.services.embedding_service import EmbeddingService
from backend.services.types import CandidatePayload, SpanPayload
from backend.services.text_processor import TextProcessor
from backend.services.storage_gateway import MatchGroupCreate, MatchDetailCreate
from backend.services.progress_tracker import ProgressTracker
from backend.models.detection_modes import DetectionMode, get_detection_config
from backend.core.logging import get_logger, LogEvent

logger = get_logger(__name__)


@dataclass
class ComparisonConfig:
    """Comparison configuration with detection mode support."""
    mode: DetectionMode = DetectionMode.AGGRESSIVE
    granularity: ChunkGranularity = ChunkGranularity.PARAGRAPH
    # Optional overrides
    semantic_threshold: Optional[float] = None
    final_threshold: Optional[float] = None
    top_k: Optional[int] = None
    # Match filtering strategy
    match_strategy: str = "bidirectional_stable"  # "all", "best_per_source", "bidirectional_stable", "bidirectional_relaxed"
    relaxed_threshold_ratio: float = 0.95  # For bidirectional_relaxed mode


@singleton
class ComparisonService(BaseService):
    """Executes pairwise comparison flow leveraging similarity pipeline."""

    def _initialize(self) -> None:
        self.orchestrator = DetectionOrchestrator()
        self.embedding_service = EmbeddingService()
        # Cross-encoder is mandatory
        self.cross_encoder_service = CrossEncoderService()
        self.text_processor = TextProcessor()
        self.progress_tracker = ProgressTracker()

    async def run_pair(self, pair_id: int, config: ComparisonConfig, progress_task_id: Optional[str] = None) -> None:
        self._ensure_initialized()

        logger.info(
            LogEvent.DETECTION_STARTED,
            pair_id=pair_id,
            mode=config.mode,
            granularity=config.granularity
        )

        try:
            pair = await self.orchestrator.update_pair_status(pair_id, status=ComparePairStatus.RUNNING)

            # 记录文档信息
            logger.info(
                "pair_documents_loaded",
                pair_id=pair_id,
                left_document_id=pair.left_document_id,
                right_document_id=pair.right_document_id
            )

            left_chunks = await self.orchestrator.fetch_chunks(pair.left_document_id)
            right_chunks = await self.orchestrator.fetch_chunks(pair.right_document_id)

            logger.info(
                "chunks_fetched",
                pair_id=pair_id,
                left_chunks_count=len(left_chunks),
                right_chunks_count=len(right_chunks)
            )

            if progress_task_id:
                await self.progress_tracker.update_progress(
                    progress_task_id,
                    message=f"Fetched chunks: {len(left_chunks)} left, {len(right_chunks)} right"
                )

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

            # Get detection mode configuration
            mode_config = get_detection_config(
                config.mode,
                config.semantic_threshold,
                config.final_threshold,
                config.top_k
            )

            # All modes now use aggressive pipeline
            pipeline = AggressiveSimilarityPipeline(config=mode_config.aggressive_config)

            logger.info(
                "pipeline_starting",
                pair_id=pair_id,
                pipeline="aggressive",
                left_chunks=len(target_left),
                right_chunks=len(target_right),
                total_comparisons=len(target_left) * len(target_right)
            )

            candidate_states = await pipeline.run(
                plan_id=pair.id,
                left_chunks=target_left,
                right_chunks=target_right,
                embedding_service=self.embedding_service,
                cross_encoder_service=self.cross_encoder_service,
            )

            logger.info(
                "pipeline_completed",
                pair_id=pair_id,
                candidates_found=len(candidate_states)
            )

            if progress_task_id:
                await self.progress_tracker.update_progress(
                    progress_task_id,
                    message=f"Found {len(candidate_states)} similarity candidates"
                )

            # Build aggregator with match strategy
            use_best_match = config.match_strategy == "best_per_source"
            aggregator = self._build_aggregator(left_map, right_map, best_match_only=use_best_match)
            # Get threshold from mode config
            threshold = mode_config.aggressive_config.final_threshold

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
                        cross_score=state.cross_score,
                        spans=state.spans or self._build_full_spans(state.payload, left_map, right_map),
                    ),
                )

            results = aggregator.finalize()

            # Apply bidirectional filtering if requested
            if config.match_strategy in ["bidirectional_stable", "bidirectional_relaxed"]:
                from backend.services.bidirectional_match_filter import BidirectionalMatchFilter

                filter = BidirectionalMatchFilter(results)
                if config.match_strategy == "bidirectional_stable":
                    results = filter.get_stable_matches()
                else:  # bidirectional_relaxed
                    results = filter.get_relaxed_matches(config.relaxed_threshold_ratio)

                # Log filtering statistics
                stats = filter.get_statistics()
                logger.info(
                    "bidirectional_filtering_applied",
                    pair_id=pair.id,
                    strategy=config.match_strategy,
                    **stats
                )

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

            if progress_task_id:
                await self.progress_tracker.update_progress(
                    progress_task_id,
                    message=f"Saved {len(group_payloads)} match groups"
                )

            metrics = self._build_metrics(results)
            logger.info(
                LogEvent.DETECTION_COMPLETED,
                pair_id=pair_id,
                match_groups=len(group_payloads),
                match_details=len(detail_payloads),
                top_score=metrics.get("top_score", 0),
                group_count=metrics.get("group_count", 0)
            )

            await self.orchestrator.update_pair_status(
                pair_id,
                status=ComparePairStatus.COMPLETED,
                metrics=metrics,
            )
        except Exception as exc:
            await self.orchestrator.update_pair_status(
                pair_id,
                status=ComparePairStatus.FAILED,
                metrics={"error": str(exc)},
            )
            logger.error(
                LogEvent.DETECTION_FAILED,
                pair_id=pair_id,
                error=str(exc),
                exc_info=True
            )
            raise

    def _determine_final_score(self, state) -> Optional[float]:
        """Prefer fused scores but fall back to strongest available signal."""
        for score in (
            state.final_score,
            state.cross_score,
            state.semantic_score,
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
        best_match_only: bool = False,
    ) -> MatchAggregator:
        left_lookup = self._build_parent_lookup(left_map)
        right_lookup = self._build_parent_lookup(right_map)
        return MatchAggregator(
            left_map=left_map,
            right_map=right_map,
            left_lookup=left_lookup,
            right_lookup=right_lookup,
            best_match_only=best_match_only,
        )

    def _build_parent_lookup(self, chunks: Dict[int, DocumentChunk]) -> Dict[int, int]:
        """改进的父级查找逻辑"""
        lookup: Dict[int, int] = {}

        # 建立段落位置索引
        paragraphs = [chunk for chunk in chunks.values() if chunk.chunk_type == ChunkGranularity.PARAGRAPH]
        sentences = [chunk for chunk in chunks.values() if chunk.chunk_type == ChunkGranularity.SENTENCE]

        paragraph_spans = [
            (p.id, p.start_pos, p.end_pos)
            for p in sorted(paragraphs, key=lambda x: x.start_pos)
        ]

        # 段落映射到自己
        for para_id, _, _ in paragraph_spans:
            lookup[para_id] = para_id

        for sentence in sentences:
            # 1. 优先使用预设的parent_chunk_id
            if sentence.parent_chunk_id:
                # 验证parent_chunk_id是否有效
                if any(para_id == sentence.parent_chunk_id for para_id, _, _ in paragraph_spans):
                    lookup[sentence.id] = sentence.parent_chunk_id
                    continue

            # 2. 使用二分查找快速定位包含句子的段落
            para_id = self._binary_search_paragraph(
                sentence.start_pos,
                sentence.end_pos,
                paragraph_spans
            )

            if para_id:
                lookup[sentence.id] = para_id
            else:
                # 3. 降级策略：找最近的段落
                lookup[sentence.id] = self._find_nearest_paragraph(
                    sentence,
                    paragraph_spans
                )

        return lookup

    def _binary_search_paragraph(
        self,
        sent_start: int,
        sent_end: int,
        paragraph_spans: List[Tuple[int, int, int]]
    ) -> Optional[int]:
        """使用二分查找快速定位包含句子的段落"""
        left, right = 0, len(paragraph_spans) - 1

        while left <= right:
            mid = (left + right) // 2
            para_id, para_start, para_end = paragraph_spans[mid]

            if para_start <= sent_start and para_end >= sent_end:
                return para_id
            elif para_end < sent_start:
                left = mid + 1
            else:
                right = mid - 1

        return None

    def _find_nearest_paragraph(
        self,
        sentence: DocumentChunk,
        paragraph_spans: List[Tuple[int, int, int]]
    ) -> int:
        """找最近的段落作为降级策略"""
        if not paragraph_spans:
            return sentence.id

        # 计算距离并找到最近的段落
        min_distance = float('inf')
        nearest_id = sentence.id

        for para_id, para_start, para_end in paragraph_spans:
            # 计算句子中心到段落中心的距离
            sent_center = (sentence.start_pos + sentence.end_pos) / 2
            para_center = (para_start + para_end) / 2
            distance = abs(sent_center - para_center)

            if distance < min_distance:
                min_distance = distance
                nearest_id = para_id

        return nearest_id

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
