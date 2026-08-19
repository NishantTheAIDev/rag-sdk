"""Tests for text splitting primitives."""

from __future__ import annotations

from rag_sdk.chunking.text import apply_overlap, recursive_split, whitespace_tokens


def test_whitespace_tokens_track_offsets() -> None:
    tokens = whitespace_tokens("one two  three")
    assert [t for t, _s, _e in tokens] == ["one", "two", "three"]
    assert tokens[0] == ("one", 0, 3)
    assert tokens[2] == ("three", 9, 14)


def test_short_text_is_single_piece() -> None:
    assert recursive_split("hello world", ["\n\n"], 512) == [("hello world", 0, 11)]


def test_split_by_separator() -> None:
    text = "aaa\n\nbbb\n\nccc"
    pieces = recursive_split(text, ["\n\n"], 8)
    assert [p[0] for p in pieces] == ["aaa\n\nbbb", "\n\nccc"]
    assert pieces[0][1:] == (0, 8)
    assert pieces[1][1:] == (8, 13)


def test_fallback_to_next_separator() -> None:
    text = "aa bb cc"
    pieces = recursive_split(text, ["\n\n", " "], 4)
    assert [p[0] for p in pieces] == ["aa ", "bb ", "cc"]
    assert sum(len(p[0]) for p in pieces) == len(text)


def test_fallback_to_hard_split() -> None:
    text = "abcdefghij"
    pieces = recursive_split(text, [], 4)
    assert [p[0] for p in pieces] == ["abcd", "efgh", "ij"]
    assert pieces[-1][1:] == (8, 10)


def test_no_separator_present_uses_next() -> None:
    text = "abcdefghij"
    pieces = recursive_split(text, ["\n\n", ""], 4)
    assert [p[0] for p in pieces] == ["abcd", "efgh", "ij"]


def test_reconstruction_without_overlap() -> None:
    text = "The quick brown fox. " * 30
    pieces = recursive_split(text, ["\n\n", "\n", ". ", " "], 64)
    joined = "".join(p[0] for p in pieces)
    assert joined == text


def test_apply_overlap_extends_pieces() -> None:
    pieces = [("abcdef", 0, 6), ("ghijkl", 6, 12)]
    overlapped = apply_overlap(pieces, 3)
    assert overlapped[0] == ("abcdef", 0, 6)
    assert overlapped[1] == ("defghijkl", 3, 12)


def test_apply_overlap_zero_is_noop() -> None:
    pieces = [("abc", 0, 3), ("def", 3, 6)]
    assert apply_overlap(pieces, 0) == pieces


def test_apply_overlap_capped_by_previous_length() -> None:
    pieces = [("ab", 0, 2), ("cdef", 2, 6)]
    overlapped = apply_overlap(pieces, 5)
    assert overlapped[1] == ("abcdef", 0, 6)