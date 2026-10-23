"""Split text into overlapping passages, returned as character offsets into the original text.

Offsets matter: a passage is always an exact slice of its source, so a quote can be checked
character for character against the document.
"""
import re

_WORD = re.compile(r"\S+")


def chunk_spans(text: str, max_words: int = 300, overlap_words: int = 40) -> list[tuple[int, int]]:
    if max_words <= overlap_words:
        raise ValueError("max_words must be larger than overlap_words")
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        return []
    step = max_words - overlap_words
    out: list[tuple[int, int]] = []
    for start in range(0, len(spans), step):
        end = min(start + max_words, len(spans))
        out.append((spans[start][0], spans[end - 1][1]))
        if end == len(spans):
            break
    return out
