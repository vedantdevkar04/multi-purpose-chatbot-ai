"""Request and response shapes.

These are the contract with the .NET API. Changing a field name here breaks the
caller silently — Pydantic will simply not populate what it cannot find — so
treat this module as a published interface rather than internal detail.
"""

from pydantic import BaseModel, Field


class Page(BaseModel):
    """Text from one page. `number` is 1-based, or None for formats without pages."""

    number: int | None
    text: str


class ExtractResponse(BaseModel):
    pages: list[Page]
    page_count: int | None = None

    # Reported rather than inferred by the caller, because "no text" has a
    # specific and common cause worth naming: a PDF of scanned images has no
    # text layer at all. Silently ingesting an empty document would leave an
    # admin wondering why their bot knows nothing about it.
    is_empty: bool


class ChunkRequest(BaseModel):
    pages: list[Page]

    # Defaults live in the .NET options and are sent explicitly, so there is one
    # place chunk sizing is configured rather than two that can disagree.
    chunk_size_chars: int = Field(default=1000, ge=200, le=8000)
    overlap_chars: int = Field(default=150, ge=0, le=2000)


class Chunk(BaseModel):
    ordinal: int
    page_number: int | None
    content: str
    char_count: int


class ChunkResponse(BaseModel):
    chunks: list[Chunk]


class IngestResponse(BaseModel):
    """Extraction and chunking in one round trip, which is the common case."""

    pages: list[Page]
    page_count: int | None = None
    is_empty: bool
    chunks: list[Chunk]
    full_text: str
