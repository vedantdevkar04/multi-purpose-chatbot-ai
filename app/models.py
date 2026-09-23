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

    # Populated when the caller asked for embeddings. None when it did not, or
    # when embedding failed — a chunk without a vector is still worth storing,
    # because it can be embedded later, whereas losing the text would mean
    # re-reading the source file.
    embedding: list[float] | None = None


class ChunkResponse(BaseModel):
    chunks: list[Chunk]


# --- embeddings -----------------------------------------------------------


class EmbedRequest(BaseModel):
    """Texts to embed.

    A LIST, not a single string, on purpose: Ollama accepts a batch and returns
    one vector per input, so a fifty-chunk document costs one round trip rather
    than fifty.
    """

    texts: list[str] = Field(min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]

    # Returned so the caller can store it alongside each vector.
    #
    # Vectors from two different models are not comparable — each model learns
    # its own arbitrary arrangement of axes, so mixing them ranks confidently
    # and wrongly with no error anywhere. The caller records this per chunk so a
    # partly re-embedded corpus is detectable.
    model: str

    # The width of each vector. The caller's database column is vector(768), so
    # a model change that alters this must be caught rather than discovered as a
    # failed insert.
    dimensions: int


class IngestResponse(BaseModel):
    """Extraction, chunking and embedding in one round trip — the common case."""

    pages: list[Page]
    page_count: int | None = None
    is_empty: bool
    chunks: list[Chunk]
    full_text: str

    # None when embedding was not requested or was unavailable. The caller uses
    # its presence to decide whether the chunks it received carry vectors.
    embedding_model: str | None = None
