"""MinHash-based filtering for high-performance Jaccard approximation."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from backend.services.base_service import BaseService, singleton
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MinHashSignature:
    """MinHash signature for a document."""
    chunk_id: int
    signature: np.ndarray
    shingles: Optional[Set[str]] = None


@dataclass
class MinHashConfig:
    """MinHash configuration parameters."""
    shingle_size: int = 3  # Character-level 3-gram for Chinese
    num_perm: int = 120  # Number of hash permutations
    bands: int = 40  # Number of bands for LSH
    rows: int = 3  # Rows per band (bands * rows = num_perm)
    threshold: float = 0.3  # Jaccard similarity threshold
    cache_size: int = 10000  # LRU cache size for signatures
    batch_size: int = 100  # Batch processing size


class MinHashProcessor:
    """Processor for MinHash computation and similarity estimation."""

    def __init__(self, config: MinHashConfig):
        self.config = config
        self._hash_funcs = self._generate_hash_functions()
        # Cache for MinHash signatures
        self._signature_cache: Dict[int, MinHashSignature] = {}

    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """Generate hash function parameters (a, b) for MinHash."""
        # Use a large prime for modulo (must fit in uint32)
        self.prime = 4294967291  # Largest prime that fits in uint32

        # Generate random hash function parameters
        np.random.seed(42)  # For reproducibility
        a_values = np.random.randint(1, self.prime, size=self.config.num_perm)
        b_values = np.random.randint(0, self.prime, size=self.config.num_perm)

        return list(zip(a_values.tolist(), b_values.tolist()))

    def _create_shingles(self, text: str) -> Set[str]:
        """Create character-level n-grams (shingles) from text."""
        if len(text) < self.config.shingle_size:
            return {text}

        # Normalize text: remove extra spaces, lowercase
        normalized = ' '.join(text.split()).lower()

        # Create character-level shingles
        shingles = set()
        for i in range(len(normalized) - self.config.shingle_size + 1):
            shingle = normalized[i:i + self.config.shingle_size]
            # Filter out pure whitespace shingles
            if not shingle.isspace():
                shingles.add(shingle)

        return shingles

    def _hash_shingle(self, shingle: str) -> int:
        """Convert shingle to integer hash."""
        # Use SHA256 and take first 8 bytes as integer
        hash_bytes = hashlib.sha256(shingle.encode('utf-8')).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big')

    def compute_signature(self, chunk_id: int, text: str) -> MinHashSignature:
        """Compute MinHash signature for a text chunk."""
        # Check cache first
        if chunk_id in self._signature_cache:
            return self._signature_cache[chunk_id]

        # Create shingles
        shingles = self._create_shingles(text)
        if not shingles:
            # Empty signature for empty text
            signature = np.full(self.config.num_perm, self.prime, dtype=np.uint64)
            return MinHashSignature(chunk_id=chunk_id, signature=signature, shingles=shingles)

        # Convert shingles to integers
        shingle_hashes = [self._hash_shingle(s) for s in shingles]

        # Compute MinHash signature
        signature = np.full(self.config.num_perm, self.prime, dtype=np.uint64)

        for shingle_hash in shingle_hashes:
            for i, (a, b) in enumerate(self._hash_funcs):
                # Universal hash function: h(x) = (a*x + b) % prime
                # Use explicit integer conversion to avoid overflow
                hash_val = int((int(a) * int(shingle_hash) + int(b)) % self.prime)
                signature[i] = min(signature[i], hash_val)

        result = MinHashSignature(chunk_id=chunk_id, signature=signature, shingles=shingles)

        # Cache the signature
        self._signature_cache[chunk_id] = result

        return result

    def estimate_jaccard(self, sig1: MinHashSignature, sig2: MinHashSignature) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1.signature) != len(sig2.signature):
            return 0.0

        # Count matching hash values
        matches = np.sum(sig1.signature == sig2.signature)
        return float(matches) / len(sig1.signature)

    def exact_jaccard(self, sig1: MinHashSignature, sig2: MinHashSignature) -> float:
        """Compute exact Jaccard similarity (for testing/validation)."""
        if sig1.shingles is None or sig2.shingles is None:
            return 0.0

        if not sig1.shingles and not sig2.shingles:
            return 1.0

        intersection = len(sig1.shingles & sig2.shingles)
        union = len(sig1.shingles | sig2.shingles)

        return float(intersection) / union if union > 0 else 0.0


class LSHIndex:
    """Locality-Sensitive Hashing index for efficient similarity search."""

    def __init__(self, config: MinHashConfig):
        self.config = config
        self.buckets: Dict[int, Dict[int, List[int]]] = {}
        self._initialize_buckets()

    def _initialize_buckets(self):
        """Initialize LSH buckets for each band."""
        for band in range(self.config.bands):
            self.buckets[band] = {}

    def _hash_band(self, signature: np.ndarray, band_idx: int) -> int:
        """Hash a band of the signature."""
        start = band_idx * self.config.rows
        end = start + self.config.rows
        band = signature[start:end]

        # Convert band to bytes and hash
        band_bytes = band.tobytes()
        hash_val = int(hashlib.md5(band_bytes).hexdigest()[:8], 16)
        return hash_val

    def add(self, chunk_id: int, signature: MinHashSignature):
        """Add a signature to the LSH index."""
        for band_idx in range(self.config.bands):
            band_hash = self._hash_band(signature.signature, band_idx)

            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []

            self.buckets[band_idx][band_hash].append(chunk_id)

    def query(self, signature: MinHashSignature, exclude_id: Optional[int] = None) -> Set[int]:
        """Find candidate similar items using LSH."""
        candidates = set()

        for band_idx in range(self.config.bands):
            band_hash = self._hash_band(signature.signature, band_idx)

            if band_hash in self.buckets[band_idx]:
                for candidate_id in self.buckets[band_idx][band_hash]:
                    if exclude_id is None or candidate_id != exclude_id:
                        candidates.add(candidate_id)

        return candidates


@singleton
class MinHashFilterStage(BaseService):
    """MinHash-based filtering stage for similarity pipeline."""

    def _initialize(self):
        """Initialize MinHash components."""
        self.config = MinHashConfig()
        self.processor = MinHashProcessor(self.config)
        self.lsh_index = LSHIndex(self.config)
        self.logger = logger

    async def process_chunks(
        self,
        chunks: Dict[int, str],
        existing_signatures: Optional[Dict[int, MinHashSignature]] = None
    ) -> Dict[int, MinHashSignature]:
        """Process chunks to compute MinHash signatures with parallel execution."""
        self._ensure_initialized()

        signatures = existing_signatures or {}
        missing_ids = [cid for cid in chunks if cid not in signatures]

        if not missing_ids:
            return signatures

        # Process in batches for efficiency
        batch_size = self.config.batch_size
        all_tasks = []

        # Create all tasks upfront for maximum parallelism
        for i in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[i:i + batch_size]
            batch_tasks = [
                self._compute_signature_async(cid, chunks[cid])
                for cid in batch_ids
            ]
            all_tasks.extend(batch_tasks)

        # Execute all tasks in parallel
        self.logger.info(f"Computing {len(all_tasks)} MinHash signatures in parallel")
        start_time = asyncio.get_event_loop().time()

        all_signatures = await asyncio.gather(*all_tasks)

        elapsed = asyncio.get_event_loop().time() - start_time
        self.logger.info(f"Computed {len(all_signatures)} signatures in {elapsed:.2f}s")

        # Update signatures and LSH index
        for sig in all_signatures:
            signatures[sig.chunk_id] = sig
            self.lsh_index.add(sig.chunk_id, sig)

        return signatures

    async def _compute_signature_async(self, chunk_id: int, text: str) -> MinHashSignature:
        """Compute signature asynchronously."""
        # Run in thread pool to avoid blocking
        return await asyncio.to_thread(
            self.processor.compute_signature, chunk_id, text
        )

    async def find_similar_pairs(
        self,
        left_signatures: Dict[int, MinHashSignature],
        right_signatures: Dict[int, MinHashSignature],
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """Find similar pairs between two sets of documents."""
        self._ensure_initialized()

        threshold = threshold or self.config.threshold
        similar_pairs = []

        # Build LSH index for right documents
        right_lsh = LSHIndex(self.config)
        for chunk_id, sig in right_signatures.items():
            right_lsh.add(chunk_id, sig)

        # Query for each left document
        for left_id, left_sig in left_signatures.items():
            # Get candidates from LSH
            candidates = right_lsh.query(left_sig)

            # Compute actual similarities for candidates
            for right_id in candidates:
                if right_id in right_signatures:
                    right_sig = right_signatures[right_id]
                    similarity = self.processor.estimate_jaccard(left_sig, right_sig)

                    if similarity >= threshold:
                        similar_pairs.append((left_id, right_id, similarity))

        return similar_pairs

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the MinHash processor."""
        self._ensure_initialized()

        return {
            "cache_size": len(self.processor._signature_cache),
            "config": {
                "shingle_size": self.config.shingle_size,
                "num_perm": self.config.num_perm,
                "bands": self.config.bands,
                "rows": self.config.rows,
                "threshold": self.config.threshold,
                "batch_size": self.config.batch_size,
            },
            "s_curve": {
                "s=0.3": self._probability_at_similarity(0.3),
                "s=0.5": self._probability_at_similarity(0.5),
                "s=0.7": self._probability_at_similarity(0.7),
                "s=0.9": self._probability_at_similarity(0.9),
            },
            "performance": {
                "cache_hit_rate": self._get_cache_hit_rate(),
                "signatures_cached": len(self.processor._signature_cache),
                "max_cache_size": self.config.cache_size,
            }
        }

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = len(self.processor._signature_cache)
        if total == 0:
            return 0.0
        # Estimate based on cache size vs max size
        return min(total / self.config.cache_size, 1.0)

    def _probability_at_similarity(self, s: float) -> float:
        """Calculate probability of LSH match at given similarity."""
        # Probability that at least one band matches
        # P = 1 - (1 - s^r)^b
        prob_band_match = s ** self.config.rows
        prob_no_match = (1 - prob_band_match) ** self.config.bands
        return 1 - prob_no_match