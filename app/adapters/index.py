"""Hybrid index: BM25 keyword scores plus cosine vector scores, with the access filter applied inside the search.

Held in memory for the PoC. The same methods can be backed by pgvector or OpenSearch later.
"""
import math

import numpy as np

from app.schemas import DocumentRecord, Passage
from app.text import tokenize


class InMemoryIndex:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.passages: dict[str, Passage] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._terms: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._total_terms = 0

    def add_document(self, record: DocumentRecord, passages: list[Passage], vectors: np.ndarray) -> None:
        self.documents[record.id] = record
        for passage, vector in zip(passages, vectors):
            terms = tokenize(passage.text)
            self.passages[passage.id] = passage
            self._vectors[passage.id] = vector
            self._terms[passage.id] = terms
            self._total_terms += len(terms)
            for term in set(terms):
                self._df[term] = self._df.get(term, 0) + 1

    def _allowed(self, passage_id: str, levels: set[str]) -> bool:
        return self.passages[passage_id].access_level in levels

    def keyword(self, query: str, levels: set[str], k: int) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        n = len(self.passages)
        if not query_terms or n == 0:
            return []
        average_length = self._total_terms / n
        scores: dict[str, float] = {}
        for passage_id, terms in self._terms.items():
            if not self._allowed(passage_id, levels):
                continue
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in set(query_terms):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * len(terms) / average_length))
            if score > 0:
                scores[passage_id] = score
        return sorted(scores.items(), key=lambda item: -item[1])[:k]

    def vector(self, query_vector: np.ndarray, levels: set[str], k: int) -> list[tuple[str, float]]:
        scored = [
            (passage_id, float(np.dot(query_vector, vector)))
            for passage_id, vector in self._vectors.items()
            if self._allowed(passage_id, levels)
        ]
        return sorted(scored, key=lambda item: -item[1])[:k]
