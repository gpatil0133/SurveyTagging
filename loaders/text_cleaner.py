"""Centralized text cleaning: HTML stripping, piping marker extraction, whitespace normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Regex patterns compiled once at module level
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NBSP_RE = re.compile(r"&nbsp;?")
_HTML_ENTITIES_RE = re.compile(r"&[a-zA-Z]+;")
_PIPING_MARKER_RE = re.compile(r"\[\[DIR_\d+_\d+_\d+\]\]")
_QUESTION_PIPING_RE = re.compile(r"\[\[Q\d+\]\]")
_PLACEHOLDER_RE = re.compile(r"\[(?:company|product|service|brand)[^\]]*\]", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass
class CleanedText:
    """Result of text cleaning, preserving extracted metadata."""

    cleaned: str
    raw: str
    piping_markers: list[str] = field(default_factory=list)
    has_piping: bool = False
    had_html: bool = False
    placeholders: list[str] = field(default_factory=list)


def clean_text(raw: str | None) -> CleanedText:
    """Clean HTML, extract piping markers, normalize whitespace.

    Args:
        raw: Raw text from survey structure (may contain HTML, piping markers, etc.)

    Returns:
        CleanedText with cleaned text and extracted metadata.
    """
    if not raw:
        return CleanedText(cleaned="", raw="")

    text = raw
    had_html = bool(_HTML_TAG_RE.search(text))

    # Extract piping markers before cleaning
    piping_markers = _PIPING_MARKER_RE.findall(text)
    piping_markers += _QUESTION_PIPING_RE.findall(text)

    # Extract placeholders like [company / product / service / brand]
    placeholders = _PLACEHOLDER_RE.findall(text)

    # Strip HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # Replace HTML entities
    text = _NBSP_RE.sub(" ", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = _HTML_ENTITIES_RE.sub(" ", text)

    # Remove piping markers from display text
    text = _PIPING_MARKER_RE.sub("", text)
    text = _QUESTION_PIPING_RE.sub("", text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return CleanedText(
        cleaned=text,
        raw=raw,
        piping_markers=piping_markers,
        has_piping=len(piping_markers) > 0,
        had_html=had_html,
        placeholders=placeholders,
    )


def clean_answer_text(raw: str | None) -> str:
    """Lightweight cleaning for answer option text (no piping extraction needed)."""
    if not raw:
        return ""
    text = _HTML_TAG_RE.sub(" ", raw)
    text = _NBSP_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
