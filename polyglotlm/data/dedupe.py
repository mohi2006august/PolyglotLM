"""Exact and near-duplicate removal with MinHash + LSH banding.

Documents are streamed; the first occurrence is kept and later duplicates are dropped.
Hashes are seeded and independent of PYTHONHASHSEED, so a rerun drops the same documents.
"""

from __future__ import annotations

import hashlib
import zlib
from collections import defaultdict

import numpy as np

_PRIME = np.uint64(4_294_967_311)  # smallest prime above 2**32


def content_hash(text: str) -> str:
    """Hash of the whitespace- and case-insensitive content, for exact duplicates."""
    canonical = " ".join(text.casefold().split())
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()


def shingles(text: str, size: int) -> np.ndarray:
    """Unique 32-bit hashes of the character n-grams of `text`."""
    canonical = " ".join(text.casefold().split())
    if len(canonical) <= size:
        grams = {canonical}
    else:
        grams = {canonical[i : i + size] for i in range(len(canonical) - size + 1)}
    return np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64)


class MinHashDeduper:
    """Streaming near-duplicate detector.

    `bands * rows` must equal `num_perm`. A document is a duplicate when it shares at least one
    LSH band with an earlier document and their signatures agree on >= `threshold` positions
    (an unbiased estimate of the Jaccard similarity of their shingle sets).
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 64,
        bands: int = 16,
        shingle_size: int = 5,
        seed: int = 0,
    ) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold = threshold
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_size = shingle_size
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2**31, size=num_perm, dtype=np.uint64)[:, None]
        self._b = rng.integers(0, 2**31, size=num_perm, dtype=np.uint64)[:, None]
        self._exact: set[str] = set()
        self._signatures: list[np.ndarray] = []
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)

    def signature(self, text: str) -> np.ndarray:
        hashed = (self._a * shingles(text, self.shingle_size)[None, :] + self._b) % _PRIME
        return hashed.min(axis=1)

    def _band_keys(self, signature: np.ndarray) -> list[tuple[int, bytes]]:
        return [
            (band, signature[band * self.rows : (band + 1) * self.rows].tobytes())
            for band in range(self.bands)
        ]

    def is_duplicate(self, text: str) -> bool:
        """True if `text` duplicates an earlier document; otherwise remember it and return False."""
        digest = content_hash(text)
        if digest in self._exact:
            return True
        signature = self.signature(text)
        keys = self._band_keys(signature)
        candidates = {idx for key in keys for idx in self._buckets.get(key, ())}
        if any(np.mean(self._signatures[idx] == signature) >= self.threshold for idx in candidates):
            return True
        self._exact.add(digest)
        self._signatures.append(signature)
        for key in keys:
            self._buckets[key].append(len(self._signatures) - 1)
        return False
