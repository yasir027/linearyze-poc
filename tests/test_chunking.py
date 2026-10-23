import pytest

from app.chunking import chunk_spans


def test_short_text_is_one_passage_that_is_an_exact_slice():
    text = "A short paragraph with only a few words."
    spans = chunk_spans(text, max_words=300, overlap_words=40)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == text


def test_long_text_is_split_with_overlap_and_every_passage_is_an_exact_slice():
    text = " ".join(f"word{i}" for i in range(1000))
    spans = chunk_spans(text, max_words=300, overlap_words=40)
    passages = [text[s:e] for s, e in spans]

    assert len(passages) > 1
    for passage in passages:
        assert passage in text
    # consecutive passages share the overlap words
    first_words = passages[0].split()
    second_words = passages[1].split()
    assert first_words[-40:] == second_words[:40]


def test_empty_text_gives_no_passages():
    assert chunk_spans("   \n  ") == []


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_spans("some text", max_words=10, overlap_words=10)
