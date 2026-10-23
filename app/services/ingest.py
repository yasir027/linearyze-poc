"""Upload path: read a file, split it into passages, embed them and store them in the index.

Idempotent: uploading the same bytes again returns the existing document and adds nothing.
"""
import hashlib

from app.adapters.embedder import Embedder
from app.adapters.index import InMemoryIndex
from app.chunking import chunk_spans
from app.parsing import extract_sections
from app.schemas import DocumentInfo, DocumentRecord, Passage

MAX_WORDS = 300
OVERLAP_WORDS = 40


class EmptyDocument(ValueError):
    pass


def info_of(record: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(id=record.id, name=record.name, access_level=record.access_level, passage_count=record.passage_count)


class IngestService:
    def __init__(self, index: InMemoryIndex, embedder: Embedder):
        self._index = index
        self._embedder = embedder

    def ingest(self, filename: str, content: bytes, access_level: str = "public") -> DocumentInfo:
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        existing = self._index.documents.get(doc_id)
        if existing is not None:
            return info_of(existing)

        passages: list[Passage] = []
        for section in extract_sections(filename, content):
            for start, end in chunk_spans(section.text, MAX_WORDS, OVERLAP_WORDS):
                passages.append(
                    Passage(
                        id=f"{doc_id}-{len(passages):04d}",
                        document_id=doc_id,
                        document_name=filename,
                        location=section.location,
                        text=section.text[start:end],
                        access_level=access_level,
                    )
                )
        if not passages:
            raise EmptyDocument(filename)

        vectors = self._embedder.embed([p.text for p in passages])
        record = DocumentRecord(
            id=doc_id,
            name=filename,
            access_level=access_level,
            passage_count=len(passages),
            content=content,
        )
        self._index.add_document(record, passages, vectors)
        return info_of(record)
