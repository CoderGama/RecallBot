"""Conservative text preprocessing that preserves facts, negation, and entities."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def preprocess_message(text: str | None) -> dict:
    """Normalize a user message without stripping meaning needed for NLI.

    Negation words, dates, numbers, and names are kept. Punctuation is
    retained except for repeated decorative characters.
    """
    raw = "" if text is None else str(text)
    original = raw
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = _CTRL.sub(" ", normalized)
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    empty = not normalized
    lowered = normalized.lower()
    return {
        "original": original,
        "text": normalized,
        "lower": lowered,
        "empty": empty,
        "too_short": len(normalized) < 2,
        "valid": (not empty) and len(normalized) < 4000,
    }
