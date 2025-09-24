"""Pipeline metrics collection for performance monitoring."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    execution_time: float = 0.0
    candidate_count_before: int = 0
    candidate_count_after: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def reduction_rate(self) -> float:
        """Calculate candidate reduction rate."""
        if self.candidate_count_before == 0:
            return 0.0
        return 1.0 - (self.candidate_count_after / self.candidate_count_before)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


@dataclass
class PipelineMetrics:
    """Aggregated metrics for entire pipeline execution."""
    pipeline_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    total_candidates_initial: int = 0
    total_candidates_final: int = 0

    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def total_reduction_rate(self) -> float:
        """Calculate overall candidate reduction rate."""
        if self.total_candidates_initial == 0:
            return 0.0
        return 1.0 - (self.total_candidates_final / self.total_candidates_initial)

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls across all stages."""
        return sum(stage.api_calls for stage in self.stages.values())

    @property
    def total_cache_hits(self) -> int:
        """Calculate total cache hits across all stages."""
        return sum(stage.cache_hits for stage in self.stages.values())

    @property
    def total_cache_misses(self) -> int:
        """Calculate total cache misses across all stages."""
        return sum(stage.cache_misses for stage in self.stages.values())

    @property
    def overall_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total = self.total_cache_hits + self.total_cache_misses
        if total == 0:
            return 0.0
        return self.total_cache_hits / total

    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "pipeline_id": self.pipeline_id,
            "total_execution_time": round(self.total_execution_time, 3),
            "total_candidates_initial": self.total_candidates_initial,
            "total_candidates_final": self.total_candidates_final,
            "total_reduction_rate": round(self.total_reduction_rate, 3),
            "total_api_calls": self.total_api_calls,
            "overall_cache_hit_rate": round(self.overall_cache_hit_rate, 3),
            "stages": {
                name: {
                    "execution_time": round(stage.execution_time, 3),
                    "candidates_before": stage.candidate_count_before,
                    "candidates_after": stage.candidate_count_after,
                    "reduction_rate": round(stage.reduction_rate, 3),
                    "api_calls": stage.api_calls,
                    "cache_hit_rate": round(stage.cache_hit_rate, 3),
                }
                for name, stage in self.stages.items()
            }
        }


class MetricsCollector:
    """Collects and aggregates pipeline metrics."""

    def __init__(self):
        self.metrics_history: List[PipelineMetrics] = []
        self.current_metrics: Optional[PipelineMetrics] = None

    def start_pipeline(self, pipeline_id: str) -> None:
        """Start tracking a new pipeline execution."""
        self.current_metrics = PipelineMetrics(pipeline_id=pipeline_id)
        logger.info(f"Started metrics collection for pipeline {pipeline_id}")

    def start_stage(self, stage_name: str, candidate_count: int) -> None:
        """Start tracking a stage execution."""
        if not self.current_metrics:
            return

        self.current_metrics.stages[stage_name] = StageMetrics(
            name=stage_name,
            candidate_count_before=candidate_count
        )

    def end_stage(
        self,
        stage_name: str,
        candidate_count: int,
        api_calls: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0
    ) -> None:
        """End tracking a stage execution."""
        if not self.current_metrics or stage_name not in self.current_metrics.stages:
            return

        stage = self.current_metrics.stages[stage_name]
        stage.execution_time = time.time() - self.current_metrics.start_time
        stage.candidate_count_after = candidate_count
        stage.api_calls = api_calls
        stage.cache_hits = cache_hits
        stage.cache_misses = cache_misses

    def end_pipeline(self, final_candidate_count: int) -> PipelineMetrics:
        """End tracking pipeline execution and return metrics."""
        if not self.current_metrics:
            raise ValueError("No active pipeline metrics")

        self.current_metrics.end_time = time.time()
        self.current_metrics.total_candidates_final = final_candidate_count

        # Store in history
        self.metrics_history.append(self.current_metrics)

        # Log summary
        logger.info(
            f"Pipeline {self.current_metrics.pipeline_id} completed",
            execution_time=round(self.current_metrics.total_execution_time, 3),
            reduction_rate=round(self.current_metrics.total_reduction_rate, 3),
            api_calls=self.current_metrics.total_api_calls,
            cache_hit_rate=round(self.current_metrics.overall_cache_hit_rate, 3)
        )

        metrics = self.current_metrics
        self.current_metrics = None
        return metrics

    def get_aggregated_stats(self) -> Dict[str, any]:
        """Get aggregated statistics from historical metrics."""
        if not self.metrics_history:
            return {}

        # Aggregate by stage
        stage_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "total_reduction": 0.0,
            "total_api_calls": 0,
            "total_cache_hits": 0,
            "total_cache_misses": 0,
        })

        total_pipelines = len(self.metrics_history)

        for metrics in self.metrics_history:
            for stage_name, stage in metrics.stages.items():
                stats = stage_stats[stage_name]
                stats["count"] += 1
                stats["total_time"] += stage.execution_time
                stats["total_reduction"] += stage.reduction_rate
                stats["total_api_calls"] += stage.api_calls
                stats["total_cache_hits"] += stage.cache_hits
                stats["total_cache_misses"] += stage.cache_misses

        # Calculate averages
        avg_stats = {}
        for stage_name, stats in stage_stats.items():
            count = stats["count"]
            total_cache = stats["total_cache_hits"] + stats["total_cache_misses"]
            avg_stats[stage_name] = {
                "avg_execution_time": round(stats["total_time"] / count, 3),
                "avg_reduction_rate": round(stats["total_reduction"] / count, 3),
                "avg_api_calls": round(stats["total_api_calls"] / count, 1),
                "avg_cache_hit_rate": round(stats["total_cache_hits"] / total_cache, 3) if total_cache > 0 else 0.0,
            }

        return {
            "total_pipelines": total_pipelines,
            "stage_stats": avg_stats,
            "overall_avg_execution_time": round(
                sum(m.total_execution_time for m in self.metrics_history) / total_pipelines, 3
            ),
            "overall_avg_reduction_rate": round(
                sum(m.total_reduction_rate for m in self.metrics_history) / total_pipelines, 3
            ),
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()