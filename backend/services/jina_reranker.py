"""Custom Jina AI reranker implementation that fixes pymilvus bug."""
from __future__ import annotations

import builtins
import contextlib
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation matching Jina's format."""
    score: float
    index: int
    document: dict | None = None


class JinaRerankFunction:
    """Custom Jina AI reranker client that fixes pymilvus parsing bug."""

    def __init__(self, model_name: str, api_key: str):
        """Initialize Jina reranker.

        Args:
            model_name: Jina model identifier (e.g., 'jina-reranker-v2-base-multilingual')
            api_key: Jina API key for authentication
        """
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = "https://api.jina.ai/v1/rerank"

        # Use synchronous HTTP client
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
    def _make_request(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        """Make rerank request to Jina API (synchronous)."""
        try:
            # Build request payload for Jina API
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
            }

            logger.debug(
                "Making Jina rerank request",
                model=self.model_name,
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                num_docs=len(documents),
                top_n=payload["top_n"]
            )

            # Make HTTP request to Jina API
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            results = []

            # Handle different response formats from Jina API
            raw_results = data.get("results", [])

            for item in raw_results:
                # Handle both object and string formats
                if isinstance(item, dict):
                    # Standard format: {"document": {"text": "..."}, "relevance_score": 0.9, "index": 0}
                    doc = item.get("document", {})
                    if isinstance(doc, dict):
                        doc_text = doc.get("text", "")
                    else:
                        # Fallback: document is a string
                        doc_text = str(doc)

                    results.append(RerankResult(
                        score=float(item.get("relevance_score", 0.0)),
                        index=int(item.get("index", 0)),
                        document={"text": doc_text} if doc_text else None,
                    ))
                elif isinstance(item, str):
                    # Malformed response: item is a string instead of dict
                    logger.warning(
                        "Received string instead of dict in Jina response",
                        item_preview=item[:100] + "..." if len(item) > 100 else item
                    )
                    # Skip this item
                    continue
                else:
                    logger.warning(
                        "Unexpected item type in Jina response",
                        item_type=type(item).__name__
                    )
                    continue

            logger.debug(
                "Jina rerank request completed",
                num_results=len(results),
                scores=[r.score for r in results[:3]],  # Only log first 3 scores
            )

            return results

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error in Jina rerank request",
                status_code=e.response.status_code,
                detail=e.response.text[:200],  # Truncate error detail
            )
            # Return empty results on error
            return []
        except Exception as e:
            logger.error("Unexpected error in Jina rerank request", error=str(e))
            return []

    def __call__(self, query: str, documents: list[str], top_k: int = 1) -> list[RerankResult]:
        """Synchronous interface matching pymilvus JinaRerankFunction API.

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
        with contextlib.suppress(builtins.BaseException):
            self.close()
