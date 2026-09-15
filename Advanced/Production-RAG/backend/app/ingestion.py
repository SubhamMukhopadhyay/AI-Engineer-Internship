"""
ingestion.py
------------
Multi-format document ingestion: validation + text extraction for
PDF, TXT, and Markdown files.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class IngestionError(Exception):
    """Raised for invalid, empty, or corrupt uploaded documents."""


@dataclass
class PageText:
    page_number: int  # 1-indexed; 0 = not paginated (txt/md)
    text: str


def validate_file(filename: str, raw_bytes: bytes) -> str:
    settings = get_settings()
    if not filename or "." not in filename:
        raise IngestionError(f"'{filename}' has no file extension.")
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext}'. Supported: PDF, TXT, Markdown."
        )
    if not raw_bytes:
        raise IngestionError(f"'{filename}' is empty.")
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise IngestionError(
            f"'{filename}' exceeds the {settings.max_file_size_mb}MB size limit."
        )
    return ext


def _decode_text(filename: str, raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw_bytes.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                f"'{filename}' could not be decoded (unknown text encoding)."
            ) from exc


def extract_pages(filename: str, raw_bytes: bytes) -> list[PageText]:
    """Validate and extract text. Raises IngestionError on failure."""
    ext = validate_file(filename, raw_bytes)

    if ext in (".txt", ".md", ".markdown"):
        text = _decode_text(filename, raw_bytes).strip()
        if not text:
            raise IngestionError(f"'{filename}' contains no readable text.")
        return [PageText(page_number=0, text=text)]

    # PDF
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except PdfReadError as exc:
        raise IngestionError(f"'{filename}' is not a valid/readable PDF: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise IngestionError(f"Failed to open '{filename}' as PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise IngestionError(
                f"'{filename}' is password-protected and could not be opened."
            ) from exc

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(PageText(page_number=i, text=text))

    if not pages:
        raise IngestionError(
            f"'{filename}' has no extractable text (possibly a scanned/"
            "image-only PDF, which is not OCR'd by this system)."
        )
    return pages


def detect_file_type(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    return {"pdf": "pdf", "txt": "txt", "md": "markdown", "markdown": "markdown"}.get(
        ext.lstrip("."), "unknown"
    )
