"""Question path: grade, retrieve, rerank, select passages, write the answer and attach verbatim quotes.

Grading is simplified for the PoC: payment requests are refused, and everything else is either answered
from the documents (outcome B) or returns the standard not-found message (outcome E).
"""
import re

from app.adapters.embedder import Embedder
from app.adapters.index import InMemoryIndex
from app.adapters.llm import Answerer
from app.adapters.reranker import Reranker
from app.config import Settings
from app.schemas import Passage, Quote, QueryResponse
from app.text import MESSAGES, is_arabic, normalize, tokenize

PAYMENT_ACTION = re.compile(
    r"^\s*(please\s+)?(pay|transfer|wire)\b"
    r"|\b(process|make|execute|issue|send)\s+(a\s+|the\s+)?(payment|transfer)s?\b"
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])\s+|\n+")


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (passage_id, _) in enumerate(ranking, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (k + rank)
    return [passage_id for passage_id, _ in sorted(scores.items(), key=lambda item: -item[1])]


def best_sentence(query: str, text: str) -> str:
    """Return the sentence of the passage that shares the most words with the query. Always an exact substring."""
    sentences = [s.strip() for s in SENTENCE_BREAK.split(text) if s and s.strip()]
    if not sentences:
        return text.strip()
    query_terms = set(tokenize(query))
    return max(sentences, key=lambda s: len(query_terms & set(tokenize(s))))


def verify_quotes(quotes: list[Quote], passages_by_id: dict[str, Passage]) -> None:
    for quote in quotes:
        if quote.text not in passages_by_id[quote.passage_id].text:
            raise RuntimeError("A quote is not an exact substring of its source passage")


class QueryService:
    def __init__(
        self,
        index: InMemoryIndex,
        embedder: Embedder,
        reranker: Reranker,
        answerer: Answerer,
        settings: Settings,
    ):
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self.answerer = answerer
        self._settings = settings

    def answer(self, question: str, access_levels: tuple[str, ...] = ("public",)) -> QueryResponse:
        question = question.strip()
        lang = "ar" if is_arabic(question) else "en"

        if PAYMENT_ACTION.search(normalize(question)):
            return QueryResponse(outcome="OT", message=MESSAGES["M10007"][lang])

        levels = set(access_levels)
        query_vector = self._embedder.embed([question])[0]
        keyword_hits = self._index.keyword(question, levels, self._settings.top_k)
        vector_hits = self._index.vector(query_vector, levels, self._settings.top_k)
        passage_ids = reciprocal_rank_fusion([keyword_hits, vector_hits])[: self._settings.top_k]
        candidates = [self._index.passages[pid] for pid in passage_ids]

        ranked = self._reranker.rerank(question, candidates)
        selected = [
            passage
            for passage, score in ranked
            if score >= self._settings.relevance_threshold
        ][: self._settings.top_n]
        if not selected:
            return QueryResponse(outcome="E", message=MESSAGES["M1"][lang])

        quotes = [
            Quote(
                text=best_sentence(question, p.text),
                passage_id=p.id,
                document_id=p.document_id,
                document_name=p.document_name,
                location=p.location,
            )
            for p in selected
        ]
        verify_quotes(quotes, {p.id: p for p in selected})

        if self.answerer.enabled:
            answer = self.answerer.answer(question, selected, lang)
            mode = "llm"
        else:
            answer = quotes[0].text
            mode = "extractive"
        return QueryResponse(outcome="B", answer=answer, quotes=quotes, mode=mode)
