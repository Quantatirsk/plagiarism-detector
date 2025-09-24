"""Cross-encoder reranking service with support for multiple providers."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import OrderedDict
from time import time

from backend.services.base_service import BaseService, singleton


PairText = Tuple[str, str]


@singleton
class CrossEncoderService(BaseService):
    """Produces rerank scores using the configured Jina AI reranker model with LRU caching."""

    # Cache configuration
    CACHE_SIZE = 10000  # Maximum number of cached results
    CACHE_TTL = 3600    # Time-to-live in seconds (1 hour)

    def _initialize(self) -> None:
        self.provider = self.settings.reranker_provider.lower()

        if self.provider == "jina":
            # Initialize Jina provider
            self.model = self.settings.cross_encoder_model
            self.api_key = self.settings.jina_api_key

            if not self.api_key:
                raise ValueError("Jina API key is required for cross-encoder service")

            try:
                from pymilvus.model.reranker import JinaRerankFunction
                self._reranker = JinaRerankFunction(
                    model_name=self.model,
                    api_key=self.api_key,
                )
                self.logger.info(f"Initialized Jina reranker with model: {self.model}")
            except ImportError:
                raise ImportError("pymilvus[model] is required for Jina cross-encoder service")
            except Exception as exc:
                self.logger.error("Failed to initialize Jina reranker", error=str(exc))
                raise

        elif self.provider == "openai":
            # Initialize OpenAI-compatible provider
            self.model = self.settings.reranker_model
            self.api_key = self.settings.reranker_openai_api_key or self.settings.openai_api_key
            self.base_url = self.settings.reranker_openai_base_url or self.settings.openai_base_url

            if not self.api_key:
                raise ValueError("API key is required for OpenAI reranker service")
            if not self.base_url:
                raise ValueError("Base URL is required for OpenAI reranker service")

            try:
                from backend.services.openai_reranker import OpenAIRerankFunction
                self._reranker = OpenAIRerankFunction(
                    model_name=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.logger.info(f"Initialized OpenAI reranker with model: {self.model}")
            except Exception as exc:
                self.logger.error("Failed to initialize OpenAI reranker", error=str(exc))
                raise

        else:
            raise ValueError(f"Unknown reranker provider: {self.provider}. Supported: 'jina', 'openai'")

        # Initialize LRU cache with TTL
        self._cache: OrderedDict[str, Tuple[float, float]] = OrderedDict()  # key -> (score, timestamp)
        self._cache_hits = 0
        self._cache_misses = 0

    async def score_pairs(self, pairs: Sequence[PairText]) -> List[float]:
        if not pairs:
            return []
        self._ensure_initialized()
        return await asyncio.to_thread(self._score_pairs_sync, pairs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_pairs_sync(self, pairs: Sequence[PairText]) -> List[float]:
        scores: List[float] = []
        for left, right in pairs:
            scores.append(self._score_pair_sync(left, right))
        return scores

    def _score_pair_sync(self, left: str, right: str) -> float:
        # Generate cache key
        cache_key = self._generate_cache_key(left, right)

        # Check cache first
        cached_score = self._get_from_cache(cache_key)
        if cached_score is not None:
            self._cache_hits += 1
            return cached_score

        self._cache_misses += 1

        # Call Jina API
        try:
            results = self._reranker(
                query=left or "",
                documents=[right or ""],
                top_k=1,
            )
            if results:
                score = results[0].score
                score = self._clamp_score(score)
                # Cache the result
                self._add_to_cache(cache_key, score)
                return score
            else:
                # No results returned
                return 0.0
        except Exception as exc:
            self.logger.error("Jina reranker request failed", error=str(exc))
            raise

    def _clamp_score(self, value: float) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    def _generate_cache_key(self, left: str, right: str) -> str:
        """Generate a cache key from text pair using SHA256 hash."""
        # Include provider and model in cache key to avoid cross-provider cache hits
        combined = f"{self.provider}\x00{self.model}\x00{left}\x00{right}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[float]:
        """Get score from cache if exists and not expired."""
        if key in self._cache:
            score, timestamp = self._cache[key]
            current_time = time()

            # Check if expired
            if current_time - timestamp > self.CACHE_TTL:
                del self._cache[key]
                return None

            # Move to end (LRU behavior)
            self._cache.move_to_end(key)
            return score

        return None

    def _add_to_cache(self, key: str, score: float) -> None:
        """Add score to cache with current timestamp."""
        current_time = time()

        # Remove oldest if cache is full
        if len(self._cache) >= self.CACHE_SIZE:
            # Remove oldest entry (first item in OrderedDict)
            self._cache.popitem(last=False)

        # Add new entry
        self._cache[key] = (score, current_time)

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "total_requests": total_requests,
        }
