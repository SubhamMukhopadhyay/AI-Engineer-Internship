"""
preprocessing.py
-----------------
Text cleaning applied after extraction and before chunking: normalises
whitespace, strips control characters, and collapses excessive blank
lines while preserving paragraph structure (important for chunk quality).
"""

from __future__ import annotations

import re

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MD_HEADER_HASH_RE = re.compile(r"^(#{1,6})\s*", re.MULTILINE)


def clean_text(text: str, is_markdown: bool = False) -> str:
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    if is_markdown:
        # keep header markers (they carry structural signal) but ensure
        # consistent spacing after them
        text = _MD_HEADER_HASH_RE.sub(lambda m: m.group(1) + " ", text)
    return text.strip()
