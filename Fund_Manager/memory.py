"""
Optimized Global Memory system for storing and retrieving experiment results.

Improvements over original:
1. Embedding cache to avoid recomputation
2. Pre-built embedding matrix for vectorized similarity
3. Lazy index rebuilding only when needed
4. Batch similarity computation using numpy
5. Memory limiting with automatic pruning
"""
import json
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import ollama

from constants import SYSTEM
from file_utils import safe_json_load, safe_json_save

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory_bank.json")


class GlobalMemory:
    """
    Optimized memory system for storing and querying experiment results.

    Uses embedding caching and vectorized similarity computation for
    efficient memory retrieval.
    """

    def __init__(self, model: str = "llama3"):
        """
        Initialize the memory system.

        Args:
            model: Ollama model to use for embeddings
        """
        self.model = model
        self.memory = self._load_memory()

        # Embedding cache: hash(text) -> embedding vector
        self._embedding_cache: Dict[str, List[float]] = {}

        # Pre-built embedding matrix for vectorized search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_valid: bool = False

        # Build initial cache from loaded memories
        self._rebuild_cache()

    def _load_memory(self) -> List[Dict]:
        """Load memory from disk with error handling."""
        data = safe_json_load(MEMORY_FILE, default=[])
        if not isinstance(data, list):
            print("[GlobalMemory] Warning: Invalid memory format, starting fresh")
            return []
        return data

    def _save_memory(self) -> None:
        """Save memory to disk using safe JSON operations."""
        # Limit memory size before saving
        max_entries = SYSTEM.MAX_MEMORY_ENTRIES
        if len(self.memory) > max_entries:
            # Keep most recent entries
            self.memory = self.memory[-max_entries:]

        safe_json_save(MEMORY_FILE, self.memory)

    def _get_text_hash(self, text: str) -> str:
        """Get a hash of text for cache lookup."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Get embedding for text with optional caching.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector
        """
        if not text:
            return self._zero_vector()

        # Check cache first
        text_hash = self._get_text_hash(text)
        if use_cache and text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            embedding = response["embedding"]

            # Cache the result
            self._embedding_cache[text_hash] = embedding

            return embedding

        except Exception as e:
            print(f"[GlobalMemory] Embedding error: {e}")
            return self._zero_vector()

    def _zero_vector(self, dim: int = 1024) -> List[float]:
        """Return a zero vector of the standard embedding dimension."""
        return [0.0] * dim

    def _rebuild_cache(self) -> None:
        """Rebuild the embedding cache from loaded memories."""
        for entry in self.memory:
            summary = entry.get("summary", "")
            embedding = entry.get("embedding")

            if summary and embedding:
                text_hash = self._get_text_hash(summary)
                self._embedding_cache[text_hash] = embedding

    def _rebuild_matrix(self) -> None:
        """
        Build a numpy matrix of all memory embeddings for vectorized search.

        Called lazily before queries to ensure matrix is up-to-date.
        """
        if self._matrix_valid:
            return

        if not self.memory:
            self._embedding_matrix = None
            self._matrix_valid = True
            return

        # Collect all embeddings
        embeddings = []
        for entry in self.memory:
            emb = entry.get("embedding")
            if emb:
                embeddings.append(emb)
            else:
                # Use zero vector for entries without embeddings
                embeddings.append(self._zero_vector())

        # Build matrix (n_memories x embedding_dim)
        self._embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Pre-compute norms for faster similarity
        self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)

        self._matrix_valid = True

    def _invalidate_matrix(self) -> None:
        """Mark the embedding matrix as needing rebuild."""
        self._matrix_valid = False

    def add_experience(self, summary: str, outcome: str, metrics: Dict[str, Any]) -> None:
        """
        Store an experiment result in memory.

        Args:
            summary: Brief description of what was tested
            outcome: "SUCCESS" or "FAILURE"
            metrics: Dictionary of numerical metrics
        """
        embedding = self._get_embedding(summary)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "outcome": outcome,
            "metrics": metrics,
            "embedding": embedding
        }

        self.memory.append(entry)
        self._invalidate_matrix()  # Matrix needs rebuild
        self._save_memory()

    def query_memory(
        self,
        query_text: str,
        threshold: float = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Query memory for relevant past experiences.

        Uses vectorized cosine similarity for efficient search.

        Args:
            query_text: Text to search for
            threshold: Minimum similarity threshold (default from constants)
            top_k: Maximum number of results to return

        Returns:
            List of relevant memory entries
        """
        if not self.memory:
            return []

        threshold = threshold or SYSTEM.MEMORY_SIMILARITY_THRESHOLD

        # Get query embedding
        query_vec = np.array(self._get_embedding(query_text), dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        # Ensure matrix is built
        self._rebuild_matrix()

        if self._embedding_matrix is None or len(self._embedding_matrix) == 0:
            return []

        # Vectorized cosine similarity
        # similarity = (M @ q) / (|M| * |q|)
        dot_products = self._embedding_matrix @ query_vec
        similarities = dot_products / (self._embedding_norms * query_norm + 1e-10)

        # Find indices above threshold
        above_threshold = similarities > threshold
        if not np.any(above_threshold):
            return []

        # Get top-k results
        indices = np.argsort(similarities)[::-1]  # Sort descending
        results = []

        for idx in indices:
            if similarities[idx] > threshold:
                results.append(self.memory[idx])
                if len(results) >= top_k:
                    break

        return results

    def get_success_patterns(self, top_k: int = 5) -> List[Dict]:
        """
        Get the most successful experiments.

        Returns:
            List of successful memory entries
        """
        successes = [m for m in self.memory if m.get("outcome") == "SUCCESS"]

        # Sort by Sharpe ratio if available
        successes.sort(
            key=lambda x: x.get("metrics", {}).get("sharpe", 0),
            reverse=True
        )

        return successes[:top_k]

    def get_failure_patterns(self, top_k: int = 5) -> List[Dict]:
        """
        Get the most recent failures for learning.

        Returns:
            List of failed memory entries (most recent first)
        """
        failures = [m for m in self.memory if m.get("outcome") == "FAILURE"]

        # Sort by timestamp (most recent first)
        failures.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return failures[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank.

        Returns:
            Dictionary with memory statistics
        """
        total = len(self.memory)
        successes = sum(1 for m in self.memory if m.get("outcome") == "SUCCESS")
        failures = sum(1 for m in self.memory if m.get("outcome") == "FAILURE")

        # Average Sharpe for successes
        success_sharpes = [
            m.get("metrics", {}).get("sharpe", 0)
            for m in self.memory
            if m.get("outcome") == "SUCCESS"
        ]
        avg_sharpe = np.mean(success_sharpes) if success_sharpes else 0

        return {
            "total_memories": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0,
            "avg_success_sharpe": float(avg_sharpe),
            "cache_size": len(self._embedding_cache),
            "matrix_valid": self._matrix_valid
        }

    def clear(self) -> None:
        """Clear all memories (use with caution)."""
        self.memory = []
        self._embedding_cache = {}
        self._invalidate_matrix()
        self._save_memory()

    def prune_old_entries(self, keep_count: int = None) -> int:
        """
        Remove old entries, keeping only the most recent.

        Args:
            keep_count: Number of entries to keep (default from constants)

        Returns:
            Number of entries removed
        """
        keep_count = keep_count or SYSTEM.MAX_MEMORY_ENTRIES

        if len(self.memory) <= keep_count:
            return 0

        original_count = len(self.memory)
        self.memory = self.memory[-keep_count:]
        removed = original_count - len(self.memory)

        self._invalidate_matrix()
        self._rebuild_cache()
        self._save_memory()

        return removed
