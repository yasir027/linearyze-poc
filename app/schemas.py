"""Data shapes used between layers. Dataclasses hold internal records; Pydantic models define the API."""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Passage:
    id: str
    document_id: str
    document_name: str
    location: str
    text: str
    access_level: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    name: str
    access_level: str
    passage_count: int
    content: bytes


class DocumentInfo(BaseModel):
    id: str
    name: str
    access_level: str
    passage_count: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    access_levels: list[str] = Field(default_factory=lambda: ["public"], max_length=10)


class Quote(BaseModel):
    text: str
    passage_id: str
    document_id: str
    document_name: str
    location: str


class QueryResponse(BaseModel):
    outcome: Literal["B", "E", "OT"]
    answer: str | None = None
    quotes: list[Quote] = Field(default_factory=list)
    message: str | None = None
    mode: Literal["llm", "extractive"] | None = None
