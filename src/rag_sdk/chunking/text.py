"""Text splitting primitives shared by chunkers.

All primitives return ``(text, start_char, end_char)`` pieces so that chunkers
can produce char-offset-accurate chunks.
"""

from __future__ import annotations

import re

Piece = tuple[str, int, int]


def whitespace_tokens(text: str) -> list[Piece]:
    """Tokenize ``text`` by whitespace, tracking character offsets."""
    return [(match.group(), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


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