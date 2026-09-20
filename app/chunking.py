"""Splitting extracted text into retrievable passages.

This is the highest-leverage code in the ingestion pipeline. Retrieval quality
in Phase 6 is mostly decided here, and changing it later means re-processing
every document — which is precisely why the originals are kept.

Two rules drive the implementation:

1. Split on meaning, not on arithmetic. A chunk beginning "...fee of 30%" with
   no idea what it refers to is worse than useless: it will be retrieved for
   the wrong questions and mislead the model when it is.

2. Overlap. Without it, a fact straddling a boundary —
   "the cancellation fee is" | "30% of the booking value" — is retrievable
   from neither half. Overlap costs storage and buys recall.
"""

import re

from .models import Chunk, Page

# A sentence boundary: .!? followed by whitespace. Deliberately simple —
# it mishandles "Dr. Smith" and "e.g." by splitting early, which produces a
# slightly short chunk rather than a wrong one. A proper sentence tokeniser is
# a dependency and a language assumption; this failure mode is cheap.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_pages(
    pages: list[Page],
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Chunk each page independently, numbering across the whole document.

    Per page, rather than over the concatenated document, so every chunk can
    record which page it came from. The cost is that a paragraph spanning a page
    break is split at the boundary; the benefit is citations, which need the
    page and cannot recover it afterwards.
    """
    chunks: list[Chunk] = []
    ordinal = 0

    for page in pages:
        for content in _chunk_text(page.text, chunk_size_chars, overlap_chars):
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    page_number=page.number,
                    content=content,
                    char_count=len(content),
                )
            )
            ordinal += 1

    return chunks


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []

    # Overlap must be smaller than the chunk, or each chunk would begin with the
    # whole of the previous one and the loop would never advance.
    overlap = min(overlap, max(0, size // 2))

    units = _split_into_units(text, size)

    chunks: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        if len(current) + 2 + len(unit) <= size:
            current = f"{current}\n\n{unit}"
        else:
            chunks.append(current)
            # Start the next chunk with the tail of this one, so anything near
            # the boundary appears in both.
            current = _tail(current, overlap)
            current = f"{current}\n\n{unit}" if current else unit

    if current.strip():
        chunks.append(current)

    return chunks


def _split_into_units(text: str, size: int) -> list[str]:
    """Break text into the largest pieces that are still smaller than a chunk.

    Paragraphs first, because a paragraph is the natural unit of meaning. Only
    an oversized paragraph is broken further — first into sentences, and only
    then, if a single sentence is still too long, at an arbitrary character
    boundary. Each step down is a small loss of coherence, so each is taken only
    when the one above it fails.
    """
    units: list[str] = []

    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= size:
            units.append(paragraph)
            continue

        # Too long: fall back to sentences.
        buffer = ""
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > size:
                # A single sentence longer than a whole chunk. Usually a table
                # flattened into one line, or a document with no punctuation.
                # Nothing meaningful left to split on, so split by length.
                if buffer:
                    units.append(buffer)
                    buffer = ""
                units.extend(
                    sentence[i : i + size] for i in range(0, len(sentence), size)
                )
                continue

            if not buffer:
                buffer = sentence
            elif len(buffer) + 1 + len(sentence) <= size:
                buffer = f"{buffer} {sentence}"
            else:
                units.append(buffer)
                buffer = sentence

        if buffer:
            units.append(buffer)

    return units


def _tail(text: str, overlap: int) -> str:
    """The last `overlap` characters, trimmed forward to a word boundary.

    Cutting mid-word would start the next chunk with a fragment like "ncellation",
    which is noise in the text and noise in its embedding.
    """
    if overlap <= 0 or len(text) <= overlap:
        return ""

    tail = text[-overlap:]
    space = tail.find(" ")

    return tail[space + 1 :].strip() if space != -1 else tail.strip()
