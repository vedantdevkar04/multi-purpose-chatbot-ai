"""Turning text into vectors, via Ollama.

The only module here that talks to a model. Everything else is string handling.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# nomic-embed-text produces 768-dimensional vectors, and the caller's Postgres
# column is vector(768). Changing the model means changing this AND a database
# migration — the two must not be allowed to drift, which is why the value is
# named here rather than scattered.
NOMIC_DIMENSIONS = 768


class EmbeddingError(Exception):
    """The embedding model could not be reached, or returned something unusable."""


class EmbeddingService:
    """Embeds text using an Ollama model.

    The HTTP client is INJECTED rather than created per call, for the same
    reason .NET uses a typed HttpClient: an httpx.AsyncClient owns a connection
    pool, so constructing one per request means a fresh TCP connection every
    time and sockets accumulating in TIME_WAIT under load.

    One client, created once at startup in dependencies.py, keeps connections
    to Ollama alive between calls. It also makes this class testable — a fake
    client can be passed in with no network involved.
    """

    def __init__(self, client: httpx.AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return NOMIC_DIMENSIONS

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text in the same order."""
        # Ollama rejects an empty input list, and "this document produced no
        # chunks" is a legitimate outcome rather than a failure.
        if not texts:
            return []

        payload = {"model": self._model, "input": texts}

        try:
            # "/api/embed", not the older "/api/embeddings" — that one is
            # deprecated, takes "prompt" instead of "input", handles a single
            # string, and returns {"embedding": [...]} singular.
            #
            # The whole batch goes in one call. Ollama returns one vector per
            # input, in order; looping would pay the round trip fifty times for
            # a fifty-chunk document.
            #
            # No cancellation parameter: asyncio raises CancelledError at the
            # await itself, so cancellation propagates without being threaded
            # through. Note CancelledError derives from BaseException, so the
            # except below does not swallow it.
            response = await self._client.post("/api/embed", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Almost always "Ollama is not running" or "that model was never
            # pulled", so the message names both the model and where it looked.
            raise EmbeddingError(
                f"Could not reach embedding model '{self._model}' "
                f"at {self._client.base_url}: {exc}"
            ) from exc

        try:
            vectors = response.json()["embeddings"]
        except (ValueError, KeyError) as exc:
            raise EmbeddingError(
                f"Embedding model '{self._model}' returned an unexpected response shape."
            ) from exc

        # The mapping between texts and vectors is POSITIONAL, so a short
        # response does not fail — it silently pairs chunk 30's text with chunk
        # 29's vector, and every chunk after the gap is wrong. That corruption
        # is permanent and invisible: retrieval keeps working and keeps
        # returning the wrong passage. Refuse the whole batch instead.
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Embedding model '{self._model}' returned {len(vectors)} vectors "
                f"for {len(texts)} inputs. Refusing a misaligned batch."
            )

        # Counts and dimensions only. The texts are customer document content
        # and must not reach the logs.
        logger.info(
            "Embedded %d text(s) with %s (%d dimensions)",
            len(texts),
            self._model,
            len(vectors[0]) if vectors else 0,
        )

        return vectors
