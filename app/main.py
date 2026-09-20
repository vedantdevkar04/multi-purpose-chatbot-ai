"""Chatbot AI service.

Extraction, chunking, and later embeddings. Everything here is a pure function
of its input.

THE RULE: this service is STATELESS AND TENANT-UNAWARE. It has no database, no
authentication, and no notion of which company a document belongs to. The .NET
API owns all persistence and all tenant scoping, enforced by PostgreSQL
row-level security.

That is not an accident of scope. Tenant isolation is guaranteed in exactly one
place, and a second component able to write rows would need its own copy of
that logic — which is where the leak would eventually come from. This service
receives bytes and returns text; the caller decides whose it is.
"""

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from .chunking import chunk_pages
from .config import settings
from .extraction import UnsupportedFormatError, extract
from .models import ChunkRequest, ChunkResponse, ExtractResponse, IngestResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot AI Service",
    description="Stateless extraction and chunking for the multi-tenant chatbot.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The .NET API surfaces this so a failed dependency is visible."""
    return {"status": "ok", "service": "ai"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    chunk_size_chars: int = Form(1000),
    overlap_chars: int = Form(150),
) -> IngestResponse:
    """Extract and chunk in one round trip — the common case for an upload.

    Chunk parameters come from the caller rather than from this service's own
    config, so sizing is configured in one place (the .NET options) instead of
    two that can silently disagree.
    """
    content = await _read_upload(file)

    pages = _extract_or_400(content, file.content_type or "", file.filename or "")

    full_text = "\n\n".join(page.text for page in pages)
    is_empty = not full_text.strip()

    # No chunks from an empty document. The caller reports this to the tenant as
    # a failure with a reason, rather than recording a document that succeeded
    # and knows nothing.
    chunks = [] if is_empty else chunk_pages(pages, chunk_size_chars, overlap_chars)

    logger.info(
        "Ingested %s: %d page(s), %d chunk(s), %d chars",
        file.filename,
        len(pages),
        len(chunks),
        len(full_text),
    )

    return IngestResponse(
        pages=pages,
        page_count=len(pages) if any(p.number for p in pages) else None,
        is_empty=is_empty,
        chunks=chunks,
        full_text=full_text,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract_only(file: UploadFile = File(...)) -> ExtractResponse:
    """Extraction without chunking, for re-processing an already-stored file."""
    content = await _read_upload(file)
    pages = _extract_or_400(content, file.content_type or "", file.filename or "")
    full_text = "\n\n".join(page.text for page in pages)

    return ExtractResponse(
        pages=pages,
        page_count=len(pages) if any(p.number for p in pages) else None,
        is_empty=not full_text.strip(),
    )


@app.post("/chunk", response_model=ChunkResponse)
def chunk_only(request: ChunkRequest) -> ChunkResponse:
    """Chunking without extraction.

    This is what makes re-chunking cheap: the .NET side already stores the
    extracted text, so changing chunk size means replaying this endpoint rather
    than re-parsing every PDF.
    """
    return ChunkResponse(
        chunks=chunk_pages(request.pages, request.chunk_size_chars, request.overlap_chars)
    )


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read()

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_bytes} bytes.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty."
        )

    return content


def _extract_or_400(content: bytes, content_type: str, file_name: str):
    try:
        return extract(content, content_type, file_name)
    except UnsupportedFormatError as exc:
        # 415 rather than 400: the request is well-formed, the format is not
        # supported. The message is shown to a tenant admin, so it names the
        # formats that would work.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except Exception as exc:
        # A corrupt or encrypted PDF. Logged in full; the caller is told
        # something it can show a human.
        logger.exception("Extraction failed for %s", file_name)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The file could not be read. It may be corrupt or password-protected.",
        ) from exc
