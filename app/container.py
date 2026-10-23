"""Builds the services from settings. This is the only place that chooses concrete adapters."""
from dataclasses import dataclass

from app.adapters.embedder import HashingEmbedder, SentenceTransformerEmbedder
from app.adapters.index import InMemoryIndex
from app.adapters.llm import ClaudeAnswerer, NoLanguageModel
from app.adapters.reranker import CrossEncoderReranker, LexicalReranker
from app.config import Settings
from app.services.ingest import IngestService
from app.services.query import QueryService


@dataclass
class Services:
    index: InMemoryIndex
    ingest: IngestService
    query: QueryService


def build_services(settings: Settings) -> Services:
    index = InMemoryIndex()
    embedder = SentenceTransformerEmbedder() if settings.embedder == "bge-m3" else HashingEmbedder()
    reranker = CrossEncoderReranker() if settings.reranker == "bge" else LexicalReranker()
    answerer = (
        ClaudeAnswerer(settings.anthropic_api_key, settings.claude_model)
        if settings.llm_enabled
        else NoLanguageModel()
    )
    return Services(
        index=index,
        ingest=IngestService(index, embedder),
        query=QueryService(index, embedder, reranker, answerer, settings),
    )
