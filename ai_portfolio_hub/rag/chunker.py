"""Recursive Character Sliding-Window Text Chunker.

Splits documents cleanly along natural paragraph, sentence, and word boundaries
while maintaining sliding-window overlap to prevent semantic fragmentation.
"""

from typing import List, Dict, Any


class RecursiveCharacterChunker:
    """Production text chunker using hierarchical splitting with overlap."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str, doc_id: str = "doc_0") -> List[Dict[str, Any]]:
        """Split raw text into structured chunks with positional metadata."""
        if not text or not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.separators)
        
        # Merge small chunks to satisfy chunk_size while maintaining overlap
        merged_chunks = []
        current_chunk = []
        current_len = 0

        for piece in raw_chunks:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current_chunk:
                chunk_str = "".join(current_chunk).strip()
                if chunk_str:
                    merged_chunks.append(chunk_str)
                # Keep overlap from end of current chunk
                overlap_chars = 0
                overlap_pieces = []
                for p in reversed(current_chunk):
                    overlap_chars += len(p)
                    overlap_pieces.insert(0, p)
                    if overlap_chars >= self.chunk_overlap:
                        break
                current_chunk = overlap_pieces
                current_len = sum(len(p) for p in current_chunk)

            current_chunk.append(piece)
            current_len += piece_len

        if current_chunk:
            final_str = "".join(current_chunk).strip()
            if final_str:
                merged_chunks.append(final_str)

        # Assemble chunk metadata dicts
        structured_chunks = []
        cursor = 0
        for idx, content in enumerate(merged_chunks):
            start = text.find(content, cursor)
            if start == -1:
                start = cursor
            end = start + len(content)
            cursor = max(cursor, start + 1)

            structured_chunks.append({
                "chunk_id": f"{doc_id}_c{idx:03d}",
                "doc_id": doc_id,
                "text": content,
                "char_start": start,
                "char_end": end,
                "char_length": len(content),
                "token_estimate": max(1, len(content) // 4),
            })

        return structured_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        for i, s in enumerate(splits):
            if not s:
                continue
            # Preserve separator
            fragment = s if i == len(splits) - 1 else s + sep
            if len(fragment) <= self.chunk_size:
                final_chunks.append(fragment)
            else:
                if remaining_seps:
                    sub_splits = self._recursive_split(fragment, remaining_seps)
                    final_chunks.extend(sub_splits)
                else:
                    final_chunks.append(fragment)

        return final_chunks
