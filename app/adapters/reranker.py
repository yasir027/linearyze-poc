"""Second-pass scoring: read each candidate together with the question and score how well it answers it.

LexicalReranker is a placeholder that counts matching terms, with extra weight for numbers and a penalty
when a number in the question is missing from the passage. CrossEncoderReranker uses BAAI/bge-reranker-v2-m3,
which reads question and passage together and is much more accurate.
"""
from typing import Protocol

from app.schemas import Passage
from app.text import tokenize

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "is", "are", "was", "were", "what", "when",
    "where", "who", "how", "did", "do", "does", "and", "or", "it", "by", "me", "my", "our", "we",
}


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[Passage]) -> list[tuple[Passage, float]]: ...


def _weight(term: str) -> float:
    return 2.0 if term.isdigit() else 1.0


class LexicalReranker:
    def rerank(self, query: str, passages: list[Passage]) -> list[tuple[Passage, float]]:
        terms = {t for t in tokenize(query) if t not in _STOPWORDS}
        total = sum(_weight(t) for t in terms) or 1.0
        numbers = {t for t in terms if t.isdigit()}
        scored: list[tuple[Passage, float]] = []
        for passage in passages:
            passage_terms = set(tokenize(passage.text))
            score = sum(_weight(t) for t in terms & passage_terms) / total
            if numbers and not numbers <= passage_terms:
                score *= 0.5
            scored.append((passage, score))
        return sorted(scored, key=lambda item: -item[1])


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: list[Passage]) -> list[tuple[Passage, float]]:
        if not passages:
            return []
        scores = self._model.predict([(query, p.text) for p in passages])
        scored = [(p, float(s)) for p, s in zip(passages, scores)]
        return sorted(scored, key=lambda item: -item[1])
