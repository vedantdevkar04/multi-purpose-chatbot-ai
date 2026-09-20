# Chatbot AI Service

Extraction, chunking, and (later) embeddings for the multi-tenant support
chatbot. FastAPI, Python 3.14.

Callers:
[multi-purpose-chatbot-backend](https://github.com/vedantdevkar04/multi-purpose-chatbot-backend)

---

## The rule this service is built around

**It is stateless and tenant-unaware.** No database, no authentication, no
notion of which company a document belongs to. Bytes in, text out.

That is deliberate, not a gap. Tenant isolation in this product is enforced by
PostgreSQL row-level security, driven by a session variable that the .NET API
sets inside its own transactions. A second component able to write rows would
need its own copy of that logic, and the second copy is where the leak comes
from. One writer, one place the isolation lives.

So: the .NET API decides whose document this is. This service just reads it.

## Why Python at all

The backend is .NET and stays that way. This service exists because text
extraction is genuinely better here:

| | .NET | Python |
|---|---|---|
| Text-layer PDFs | PdfPig — fine | PyMuPDF — fine |
| Multi-column, tables | poor | **markedly better** |
| DOCX / PPTX / HTML | a library each | `unstructured` — one API |
| Scanned pages (OCR) | fiddly | pytesseract, easyocr |

Embeddings in Phase 5 will strengthen the case further.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requirements use `>=` rather than exact pins. Python 3.14 is new enough that
pinning an older package version frequently means no wheel exists and pip tries
to compile from source — which is exactly how `pydantic-core` failed here.
Pin exactly once there is a lockfile and a CI image with a fixed interpreter.

## Run

```bash
uvicorn app.main:app --port 8000 --reload
```

Interactive docs at http://localhost:8000/docs

## Endpoints

| | |
|---|---|
| `GET /health` | liveness |
| `POST /ingest` | file → pages, chunks, full text. One round trip, the common case |
| `POST /extract` | file → pages only. For re-processing a stored original |
| `POST /chunk` | pages → chunks. Re-chunking without re-parsing the PDF |

`/extract` and `/chunk` exist separately because reprocessing comes in two
kinds: chunk sizing will change (needs only the stored text), and extraction
quality will improve (needs the original file). The .NET side keeps both.

## Configuration

Environment variables, prefixed `AI_`, or a `.env` file.

| | Default | |
|---|---|---|
| `AI_MAX_UPLOAD_BYTES` | 20 MB | second line of defence; the API limits first |
| `AI_OLLAMA_BASE_URL` | `http://localhost:11434` | unused until Phase 5 |
| `AI_EMBEDDING_MODEL` | `nomic-embed-text` | unused until Phase 5 |

## Layout

```
app/
  main.py        FastAPI app and endpoints
  models.py      request/response shapes — the contract with .NET
  extraction.py  PyMuPDF and plain text
  chunking.py    paragraph-aware splitting with overlap
  config.py      settings
scripts/
  ollama_smoke_test.py   standalone Ollama check, predates this service
```

### On chunking

`chunking.py` is the highest-leverage file here — retrieval quality later is
mostly decided by it, and changing it means re-processing every document.

It splits on paragraphs first, falls back to sentences for oversized
paragraphs, and only then splits blindly by length. Each step down loses some
coherence, so each is taken only when the one above it fails.

Chunks **overlap** by default. Without overlap a fact straddling a boundary —
`"the cancellation fee is"` | `"30% of the booking value"` — is retrievable
from neither half.
