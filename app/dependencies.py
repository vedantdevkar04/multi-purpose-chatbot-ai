"""Application-lifetime resources and how endpoints get hold of them.

The HTTP client to Ollama is created once at startup and closed at shutdown,
rather than per request. httpx.AsyncClient owns a connection pool; one per
call means a new TCP connection every time and sockets piling up in TIME_WAIT.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Request

from .config import settings
from .embedding import EmbeddingService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Creates the Ollama client on startup, closes it on shutdown.

    Stored on app.state rather than in a module-level global so tests can build
    an app with a different client, and so shutdown actually closes it — a
    global would leak the connection pool.
    """
    client = httpx.AsyncClient(
        base_url=settings.ollama_base_url.rstrip("/"),
        # Generous. Embedding fifty chunks on CPU is seconds, not milliseconds,
        # and httpx defaults to five seconds — short enough to cut off a
        # perfectly healthy batch. The background worker is waiting on this,
        # not a user, so a slow answer beats a failed one.
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    app.state.embedding_service = EmbeddingService(client, settings.embedding_model)

    logger.info(
        "Embedding service ready: model=%s at %s",
        settings.embedding_model,
        settings.ollama_base_url,
    )

    try:
        yield
    finally:
        await client.aclose()


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


# Endpoints declare `service: EmbeddingServiceDep` and FastAPI supplies the
# single shared instance.
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
