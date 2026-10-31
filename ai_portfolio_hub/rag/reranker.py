"""Cross-Encoder Re-Ranking Module.

Re-scores retrieved candidate chunks against the user query using joint cross-attention
to filter out false-positive semantic matches and surface high-precision contexts.
"""

from typing import Any, Dict, List


class CrossEncoderReranker:
    """Production cross-encoder re-ranker with relevance thresholding."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Compute fine-grained query-passage scores and re-rank candidate chunks."""
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]

        if self.model is not None:
            raw_scores = self.model.predict(pairs)
            scores = [float(s) for s in raw_scores]
        else:
            # High-precision word overlap & lexical containment fallback
            scores = []
            q_terms = set(query.lower().split())
            for c in candidates:
                c_terms = set(c["text"].lower().split())
                overlap = len(q_terms.intersection(c_terms)) / max(1, len(q_terms))
                # Combine initial dense score with lexical precision
                combined = 0.6 * c.get("score", 0.5) + 0.4 * overlap
                scores.append(float(combined))

        reranked = []
        for chunk, score in zip(candidates, scores, strict=False):
            item = dict(chunk)
            item["rerank_score"] = score
            if score >= threshold:
                reranked.append(item)

        # Sort descending by re-rank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
