"""Turn text into vectors. Similar meaning should give nearby vectors.

HashingEmbedder is a dependency-free placeholder. It only matches shared words and character
fragments, so it finds keywords but not meaning. SentenceTransformerEmbedder uses BAAI/bge-m3,
which handles Arabic and English in one vector space.
"""
import hashlib
from typing import Protocol

import numpy as np

from app.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        features: list[str] = []
        for token in tokenize(text):
            features.append(token)
            padded = f"#{token}#"
            features.extend(padded[i : i + 3] for i in range(max(1, len(padded) - 2)))
        return features

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = int(hashlib.md5(feature.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
                sign = 1.0 if (digest >> 1) & 1 else -1.0
                out[row, digest % self.dim] += sign
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
