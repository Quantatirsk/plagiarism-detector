"""Cross-encoder reranking service backed by Jina AI."""
from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

from pymilvus.model.reranker import JinaRerankFunction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.base_service import BaseService, singleton


PairText = Tuple[str, str]


@singleton
class CrossEncoderService(BaseService):
    """Produces rerank scores using the configured Jina AI reranker model."""

    def _initialize(self) -> None:
        self.model = self.settings.cross_encoder_model
        self.api_key = self.settings.jina_api_key
        self._remote_available = bool(self.api_key)
        self._reranker: Optional[JinaRerankFunction] = None
        if self._remote_available and JinaRerankFunction is not None:
            try:
                self._reranker = JinaRerankFunction(
                    model_name=self.model,
                    api_key=self.api_key,
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                self.logger.warning("Failed to initialise Jina reranker", error=str(exc))
                self._remote_available = False
        else:
            if JinaRerankFunction is None:
                self.logger.warning("pymilvus[model] not installed; using TF-IDF fallback reranker")
            elif not self.api_key:
                self.logger.warning("Jina API key not configured; using TF-IDF fallback reranker")

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
        if self._remote_available and self._reranker is not None:
            try:
                results = self._reranker(
                    query=left or "",
                    documents=[right or ""],
                    top_k=1,
                )
                if results:
                    score = results[0].score
                    return self._clamp_score(score)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.logger.warning("Jina reranker request failed", error=str(exc))
                self._remote_available = False
        return self._tfidf_score(left, right)

    def _clamp_score(self, value: float) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    def _tfidf_score(self, left: str, right: str) -> float:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([left or "", right or ""])
        if matrix.shape[1] == 0:
            return 0.0
        similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
        similarity = max(min(similarity, 1.0), -1.0)
        return (similarity + 1.0) / 2.0
