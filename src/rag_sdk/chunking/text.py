"""Text splitting primitives shared by chunkers.

All primitives return ``(text, start_char, end_char)`` pieces so that chunkers
can produce char-offset-accurate chunks.
"""

from __future__ import annotations

import re

Piece = tuple[str, int, int]

# Simplified sentence split pattern - matches sentence-ending punctuation followed by
# whitespace and capital letter, or end of string. We avoid variable-width lookbehind.
_SENTENCE_END_PATTERN = re.compile(r"""[.!?]+(["']?)\s+(?=[A-Z0-9])|[.!?]+(["']?)\s*$""")

_ABBREVIATIONS = frozenset({
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "sr.", "jr.", "st.", "ave.", "blvd.", "rd.", "ln.", "ct.", "pl.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.",
    "oct.", "nov.", "dec.", "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",
    "a.m.", "p.m.", "am.", "pm.", "u.s.", "u.k.", "u.n.", "e.u.",
    "ph.d.", "m.d.", "b.a.", "m.a.", "b.s.", "m.s.", "ll.b.", "j.d.",
})


def whitespace_tokens(text: str) -> list[Piece]:
    """Tokenize ``text`` by whitespace, tracking character offsets."""
    return [(match.group(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def split_sentences(text: str) -> list[Piece]:
    """Split ``text`` into sentences using rule-based detection.

    Returns list of (sentence_text, start_char, end_char) relative to ``text``.
    Handles common abbreviations to avoid over-splitting.
    """
    if not text:
        return []

    pieces: list[Piece] = []
    last_end = 0

    for match in _SENTENCE_END_PATTERN.finditer(text):
        end = match.end()
        candidate = text[last_end:end].strip()
        if not candidate:
            last_end = end
            continue

        # Check if candidate ends with a known abbreviation
        words = candidate.split()
        if words and words[-1].lower() in _ABBREVIATIONS:
            # Likely an abbreviation, not a sentence boundary
            continue

        pieces.append((candidate, last_end, end))
        last_end = end

    # Capture any remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            pieces.append((remaining, last_end, len(text)))

    return pieces


def compute_sentence_boundaries(
    document_text: str, chunk_start: int, chunk_end: int
) -> list[tuple[int, int]]:
    """Compute sentence boundaries within a chunk, relative to the document.

    Args:
        document_text: Full document text
        chunk_start: Start character offset of chunk in document
        chunk_end: End character offset of chunk in document

    Returns:
        List of (start, end) tuples relative to document, for sentences
        that overlap with the chunk.
    """
    chunk_text = document_text[chunk_start:chunk_end]
    if not chunk_text:
        return []

    sentences = split_sentences(chunk_text)
    boundaries = []
    for _sent_text, sent_start, sent_end in sentences:
        boundaries.append((chunk_start + sent_start, chunk_start + sent_end))
    return boundaries


def recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[Piece]:
    """Split ``text`` into pieces of at most ``chunk_size`` characters.

    Splits hierarchically on ``separators``, falling back to the next
    separator when a separator does not occur, and to a hard character split
    when no separator remains.
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
    if not separators:
        return _split_by_size(text, chunk_size)
    parts = re.split(f"({re.escape(separators[0])})", text)
    if len(parts) <= 1:
        return recursive_split(text, separators[1:], chunk_size)

    pieces: list[Piece] = []
    current: list[str] = []
    current_len = 0
    current_start = 0
    offset = 0
    for part in parts:
        part_len = len(part)
        if current_len + part_len <= chunk_size:
            if not current:
                current_start = offset
            current.append(part)
            current_len += part_len
            offset += part_len
            continue
        if current:
            pieces.append(("".join(current), current_start, offset))
            current = []
            current_len = 0
        if part_len <= chunk_size:
            current.append(part)
            current_start = offset
            current_len = part_len
            offset += part_len
        else:
            sub_start = offset
            offset += part_len
            pieces.extend(
                (text, start + sub_start, end + sub_start)
                for text, start, end in recursive_split(part, separators[1:], chunk_size)
            )
    if current:
        pieces.append(("".join(current), current_start, offset))
    return pieces


def apply_overlap(pieces: list[Piece], overlap: int) -> list[Piece]:
    """Extend each piece to include the tail of the previous piece."""
    if overlap <= 0:
        return pieces
    result: list[Piece] = []
    for text, start, end in pieces:
        if result:
            prev_text, _prev_start, prev_end = result[-1]
            cut = min(overlap, len(prev_text))
            new_start = prev_end - cut
            if new_start < start:
                text = prev_text[-cut:] + text
                start = new_start
        result.append((text, start, end))
    return result


def _split_by_size(text: str, chunk_size: int) -> list[Piece]:
    return [
        (text[i : i + chunk_size], i, min(i + chunk_size, len(text)))
        for i in range(0, len(text), chunk_size)
    ]


class Tokenizer:
    """Base tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    """Simple whitespace tokenizer."""

    def encode(self, text: str) -> list[int]:
        return text.split()


try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class Cl100kBaseTokenizer(Tokenizer):
    """OpenAI cl100k_base tokenizer (GPT-3.5/4)."""

    def __init__(self) -> None:
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for cl100k_base tokenizer. "
                "Install with 'pip install tiktoken'."
            )
        self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)


def get_tokenizer(name: str) -> Tokenizer:
    """Get a tokenizer by name."""
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "cl100k_base":
        return Cl100kBaseTokenizer()
    raise ValueError(f"Unknown tokenizer: {name}")