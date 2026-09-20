"""Turning uploaded bytes into text.

This is the reason the AI service exists in Python. PDF text extraction is
genuinely hard — multi-column layouts, tables, headers repeated on every page,
reading order that does not match the order glyphs appear in the file — and
PyMuPDF handles it markedly better than the .NET options.
"""

import logging
import re

# `import pymupdf`, not the older `import fitz` — the fitz alias is deprecated
# and emits a warning on every import.
import pymupdf

from .models import Page

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPES = {"application/pdf"}
TEXT_CONTENT_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
TEXT_EXTENSIONS = (".txt", ".md", ".markdown")


class UnsupportedFormatError(Exception):
    """The file is not a format this service can read."""


def extract(content: bytes, content_type: str, file_name: str) -> list[Page]:
    """Dispatch on content type, falling back to the extension.

    Browsers are inconsistent about the content type they attach to uploads —
    a .md file frequently arrives as application/octet-stream — so the filename
    is a necessary second opinion rather than a convenience.
    """
    normalised = (content_type or "").split(";")[0].strip().lower()
    lower_name = (file_name or "").lower()

    if normalised in PDF_CONTENT_TYPES or lower_name.endswith(".pdf"):
        return _extract_pdf(content)

    if normalised in TEXT_CONTENT_TYPES or lower_name.endswith(TEXT_EXTENSIONS):
        return _extract_plain_text(content)

    raise UnsupportedFormatError(
        f"Cannot read '{file_name}' ({content_type or 'unknown type'}). "
        "Supported formats are PDF, plain text and Markdown."
    )


def _extract_pdf(content: bytes) -> list[Page]:
    pages: list[Page] = []

    # "sort=True" orders text blocks by position on the page rather than by the
    # order they happen to appear in the content stream. Without it, a
    # two-column page comes out with the columns interleaved line by line —
    # technically all the words, arranged into nonsense.
    with pymupdf.open(stream=content, filetype="pdf") as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            pages.append(Page(number=index, text=_normalise(text)))

    return pages


def _extract_plain_text(content: bytes) -> list[Page]:
    # errors="replace" rather than raising: a single bad byte in an otherwise
    # readable document should not fail the whole upload. The replacement
    # character is visible in the chunk if anyone looks.
    text = content.decode("utf-8", errors="replace")

    # No pages in a text file, so number is None. That is carried through to the
    # database as NULL rather than a fake page 1, so a future citation feature
    # can tell "page unknown" from "page one".
    return [Page(number=None, text=_normalise(text))]


def _normalise(text: str) -> str:
    """Tidy extracted text without destroying its structure.

    Paragraph breaks are load-bearing: the chunker splits on them, so collapsing
    all whitespace here would leave it with one enormous paragraph and force it
    into blind character splitting.
    """
    # Windows and old-Mac line endings to \n, so the paragraph regex below only
    # has one form to match.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trailing spaces before a newline turn "word \n" into a false word break.
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Three or more newlines to exactly two. Preserves the paragraph boundary
    # while removing the large vertical gaps PDFs produce between blocks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
