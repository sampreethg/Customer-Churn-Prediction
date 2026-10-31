"""Dense Vector Indexer for Retrieval-Augmented Generation.

Encodes document chunks into normalized dense vector embeddings and provides
efficient inner product / cosine similarity nearest neighbor search.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class DenseVectorIndex:
    """In-memory dense semantic vector index with cosine similarity search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.chunks: List[Dict[str, Any]] = []
        self._init_encoder()

    def _init_encoder(self):
        """Attempts to load sentence_transformers, with fallback to character TF-IDF hash."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        except Exception:
            # Fallback lightweight deterministic embedding generator
            self.encoder = None

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        if self.encoder is not None:
            emb = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)
        
        # Robust fallback: normalized hashed semantic feature vectors
        vectors = []
        for text in texts:
            words = text.lower().split()
            v = np.zeros(384, dtype=np.float32)
            for w in words:
                h = hash(w) % 384
                v[h] += 1.0
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v = v / norm
            vectors.append(v)
        return np.vstack(vectors)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Ingest and index a batch of structured chunks."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        new_embs = self._encode_text(texts)

        if len(self.chunks) == 0:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks using cosine similarity."""
        if len(self.chunks) == 0:
            return []

        q_emb = self._encode_text([query])[0]  # shape (dim,)
        # Cosine similarity is dot product when vectors are normalized
        sims = np.dot(self.embeddings, q_emb)

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(sims[idx])
            results.append(chunk)

        return results

    def clear(self):
        """Reset the index."""
        self.embeddings = np.empty((0, 384), dtype=np.float32)
        self.chunks.clear()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
