"""OpenAI-compatible reranker implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation matching Jina's format."""
    score: float
    index: int
    document: Optional[dict] = None


class OpenAIRerankFunction:
    """OpenAI-compatible reranker client with synchronous interface."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        """Initialize OpenAI reranker.

        Args:
            model_name: Model identifier for reranking
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/rerank"

        # Use synchronous HTTP client to avoid event loop issues
        self.client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _make_request(self, query: str, documents: List[str], top_k: int) -> List[RerankResult]:
        """Make rerank request to OpenAI-compatible endpoint (synchronous)."""
        try:
            # Build request payload
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),  # OpenAI uses top_n instead of top_k
            }

            logger.debug(
                "Making rerank request",
                model=self.model_name,
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                num_docs=len(documents),
                top_n=payload["top_n"]
            )

            # Make HTTP request (synchronous)
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            results = []

            for item in data.get("results", []):
                results.append(RerankResult(
                    score=float(item.get("relevance_score", 0.0)),
                    index=int(item.get("index", 0)),
                    document=item.get("document"),
                ))

            logger.debug(
                "Rerank request completed",
                num_results=len(results),
                scores=[r.score for r in results[:3]],  # Only log first 3 scores
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error in rerank request",
                status_code=e.response.status_code,
                detail=e.response.text[:200],  # Truncate error detail
            )
            # Return empty results on error
            return []
        except Exception as e:
            logger.error("Unexpected error in rerank request", error=str(e))
            return []

    def __call__(self, query: str, documents: List[str], top_k: int = 1) -> List[RerankResult]:
        """Synchronous interface matching Jina's API.

        Args:
            query: Query text to compare against documents
            documents: List of documents to rerank
            top_k: Number of top results to return

        Returns:
            List of RerankResult objects sorted by score (descending)
        """
        # Handle empty inputs
        if not query or not documents:
            return []

        # Call synchronous method directly
        return self._make_request(query, documents, top_k)

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass