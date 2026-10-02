"""Cross-lingual semantic search over an in-memory corpus with the distilled encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from polyglotlm.embed.encoder import TextEncoder


class EmbeddingIndex:
    """Cosine-similarity search; embeddings are L2-normalised so a dot product suffices."""

    def __init__(self, encoder: TextEncoder, texts: list[str]) -> None:
        self.encoder = encoder
        self.texts = texts
        self.vectors = encoder.encode(texts) if texts else torch.empty(0, encoder.dim)

    @classmethod
    def from_file(cls, encoder: TextEncoder, path: Path) -> EmbeddingIndex:
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls(encoder, [line.strip() for line in lines if line.strip()])

    def search(self, query: str, k: int = 5) -> list[tuple[float, str]]:
        if not self.texts:
            return []
        scores = self.encoder.encode([query])[0] @ self.vectors.T
        top = torch.topk(scores, min(k, len(self.texts)))
        return [(float(s), self.texts[i]) for s, i in zip(top.values, top.indices, strict=True)]
