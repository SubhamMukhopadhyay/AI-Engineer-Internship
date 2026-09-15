"""
extraction.py
-------------
File validation and text extraction for PDF/TXT uploads.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentError(Exception):
    """Raised for invalid, empty, or corrupt uploaded documents."""


@dataclass
class PageText:
    page_number: int  # 1-indexed; 0 means "not paginated" (txt files)
    text: str


ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def validate_file(filename: str, raw_bytes: bytes) -> str:
    """Validate extension and size. Returns the lowercase extension."""
    if not filename or "." not in filename:
        raise DocumentError(f"'{filename}' has no file extension.")
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentError(
            f"Unsupported file type '{ext}'. Only PDF and TXT are supported."
        )
    if not raw_bytes:
        raise DocumentError(f"'{filename}' is empty.")
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise DocumentError(
            f"'{filename}' is larger than the 25 MB limit."
        )
    return ext


def extract_pages(filename: str, raw_bytes: bytes) -> list[PageText]:
    """
    Extract text from a validated PDF or TXT file. Raises DocumentError
    on corrupt files or files that yield no extractable text.
    """
    ext = validate_file(filename, raw_bytes)

    if ext == ".txt":
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_bytes.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise DocumentError(
                    f"'{filename}' could not be decoded as text (unknown encoding)."
                ) from exc
        text = text.strip()
        if not text:
            raise DocumentError(f"'{filename}' contains no readable text.")
        return [PageText(page_number=0, text=text)]

    # PDF
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except PdfReadError as exc:
        raise DocumentError(f"'{filename}' is not a valid or readable PDF: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentError(f"Failed to open '{filename}' as a PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentError(
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
        raise DocumentError(
            f"'{filename}' contains no extractable text. It may be a "
            "scanned/image-only PDF, which this tool does not OCR."
        )
    return pages
