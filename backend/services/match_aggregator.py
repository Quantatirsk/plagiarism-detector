"""Aggregation helpers for grouping sentence-level matches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from backend.db.models import DocumentChunk
from backend.services.types import SpanPayload


@dataclass
class MatchState:
    final_score: Optional[float]
    semantic_score: Optional[float]
    cross_score: Optional[float]
    spans: Sequence[SpanPayload]


class MatchAggregator:
    """Aggregate sentence-level match signals into paragraph groups."""

    def __init__(
        self,
        *,
        left_map: Dict[int, DocumentChunk],
        right_map: Dict[int, DocumentChunk],
        left_lookup: Dict[int, int],
        right_lookup: Dict[int, int],
        best_match_only: bool = False,  # 新增参数：是否只保留最佳匹配
    ) -> None:
        self.left_map = left_map
        self.right_map = right_map
        self.left_lookup = left_lookup
        self.right_lookup = right_lookup
        self.best_match_only = best_match_only
        self._records: Dict[tuple[int, int], Dict[str, object]] = {}
        # 追踪每个左侧段落的最佳匹配（仅在best_match_only=True时使用）
        self._best_matches: Dict[int, tuple[int, float]] = {}

    def add(self, left_id: int, right_id: int, state: MatchState) -> None:
        left_parent = self.left_lookup.get(left_id, left_id)
        right_parent = self.right_lookup.get(right_id, right_id)

        # 如果启用最佳匹配模式
        if self.best_match_only:
            current_score = state.final_score or 0.0

            # 检查是否已有该左侧段落的匹配
            if left_parent in self._best_matches:
                best_right, best_score = self._best_matches[left_parent]

                # 如果当前分数不如已有最佳匹配，则跳过
                if current_score <= best_score:
                    return

                # 移除旧的最佳匹配记录
                old_key = (left_parent, best_right)
                if old_key in self._records:
                    del self._records[old_key]

            # 更新最佳匹配
            self._best_matches[left_parent] = (right_parent, current_score)

        key = (left_parent, right_parent)
        record = self._records.get(key)
        if not record:
            record = {
                "left_chunk_id": left_parent,
                "right_chunk_id": right_parent,
                "final_score": None,
                "semantic_score": None,
                "cross_score": None,
                "span_set": set(),
                "doc_span_set": set(),
                "match_count": 0,
                "details": [],
            }
            self._records[key] = record

        record["final_score"] = self._max_optional(record["final_score"], state.final_score)
        record["semantic_score"] = self._max_optional(record["semantic_score"], state.semantic_score)
        record["cross_score"] = self._max_optional(record["cross_score"], state.cross_score)
        record["match_count"] = int(record["match_count"]) + 1

        span_set: set = record["span_set"]  # type: ignore[assignment]
        doc_span_set: set = record["doc_span_set"]  # type: ignore[assignment]
        if state.spans:
            left_chunk = self.left_map[left_parent]
            right_chunk = self.right_map[right_parent]
            for span in state.spans:
                span_set.add((span.left_start, span.left_end, span.right_start, span.right_end))
                doc_span_set.add(
                    (
                        left_chunk.start_pos + span.left_start,
                        left_chunk.start_pos + span.left_end,
                        right_chunk.start_pos + span.right_start,
                        right_chunk.start_pos + span.right_end,
                    )
                )

        details: list = record["details"]  # type: ignore[assignment]
        details.append(
            {
                "left_chunk_id": left_id,
                "right_chunk_id": right_id,
                "final_score": state.final_score,
                "semantic_score": state.semantic_score,
                "cross_score": state.cross_score,
                "group_pair": (left_parent, right_parent),
                "spans": [
                    {
                        "left_start": span.left_start,
                        "left_end": span.left_end,
                        "right_start": span.right_start,
                        "right_end": span.right_end,
                    }
                    for span in state.spans or []
                ],
                "document_spans": [
                    {
                        "left_start": self.left_map[left_id].start_pos + span.left_start,
                        "left_end": self.left_map[left_id].start_pos + span.left_end,
                        "right_start": self.right_map[right_id].start_pos + span.right_start,
                        "right_end": self.right_map[right_id].start_pos + span.right_end,
                    }
                    for span in state.spans or []
                ],
            }
        )

    def finalize(self) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for (left_parent, right_parent), record in self._records.items():
            span_set: set = record.pop("span_set")  # type: ignore[assignment]
            doc_span_set: set = record.pop("doc_span_set")  # type: ignore[assignment]
            details = record.pop("details")  # type: ignore[assignment]
            spans = [
                SpanPayload(
                    left_start=left_start,
                    left_end=left_end,
                    right_start=right_start,
                    right_end=right_end,
                )
                for (left_start, left_end, right_start, right_end) in sorted(span_set)
            ]
            left_chunk = self.left_map[left_parent]
            right_chunk = self.right_map[right_parent]
            alignment_ratio = self._compute_alignment_ratio(spans, left_chunk, right_chunk)
            document_spans = [
                {
                    "left_start": left_start,
                    "left_end": left_end,
                    "right_start": right_start,
                    "right_end": right_end,
                }
                for (left_start, left_end, right_start, right_end) in sorted(doc_span_set)
            ]
            result = {
                "left_chunk_id": left_parent,
                "right_chunk_id": right_parent,
                "final_score": record["final_score"],
                "semantic_score": record["semantic_score"],
                "cross_score": record["cross_score"],
                "alignment_ratio": alignment_ratio,
                "span_count": len(document_spans) or len(spans),
                "match_count": record["match_count"],
                "paragraph_spans": [
                    {
                        "left_start": span.left_start,
                        "left_end": span.left_end,
                        "right_start": span.right_start,
                        "right_end": span.right_end,
                    }
                    for span in spans
                ],
                "document_spans": document_spans,
                "details": details,
            }
            self._round_numeric_fields(
                result,
                (
                    "final_score",
                    "semantic_score",
                    "cross_score",
                    "alignment_ratio",
                ),
            )
            for detail in result["details"]:
                if isinstance(detail, dict):
                    self._round_numeric_fields(
                        detail,
                        (
                            "final_score",
                            "semantic_score",
                                    "cross_score",
                        ),
                    )
            results.append(result)

        results.sort(key=self._sort_key)
        return results

    @staticmethod
    def _max_optional(current: Optional[float], value: Optional[float]) -> Optional[float]:
        if value is None:
            return current
        if current is None or value > current:
            return value
        return current

    def _compute_alignment_ratio(
        self,
        spans: Sequence[SpanPayload],
        left_chunk: DocumentChunk,
        right_chunk: DocumentChunk,
    ) -> Optional[float]:
        if not spans:
            return None
        left_overlap = self._interval_coverage(spans, side="left")
        right_overlap = self._interval_coverage(spans, side="right")
        overlap = min(left_overlap, right_overlap)
        denom = max(min(len(left_chunk.text), len(right_chunk.text)), 1)
        return overlap / denom if denom else None

    @staticmethod
    def _interval_coverage(spans: Sequence[SpanPayload], *, side: str) -> int:
        intervals: List[tuple[int, int]] = []
        for span in spans:
            if side == "left":
                intervals.append((span.left_start, span.left_end))
            else:
                intervals.append((span.right_start, span.right_end))
        if not intervals:
            return 0
        intervals.sort()
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return sum(end - start for start, end in merged)

    @staticmethod
    def _round_numeric_fields(entry: Dict[str, object], fields: Sequence[str], *, ndigits: int = 4) -> None:
        for field in fields:
            value = entry.get(field)
            if isinstance(value, float):
                entry[field] = round(value, ndigits)

    def _sort_key(self, item: Dict[str, object]) -> tuple[int, int, float]:
        left_chunk = self.left_map[item["left_chunk_id"]]
        right_chunk = self.right_map[item["right_chunk_id"]]
        left_anchor = left_chunk.start_pos
        right_anchor = right_chunk.start_pos
        score = float(item.get("final_score") or 0.0)
        return (left_anchor, right_anchor, -score)


__all__ = ["MatchAggregator", "MatchState"]
