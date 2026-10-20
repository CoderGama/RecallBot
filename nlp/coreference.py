"""Lightweight coreference fallback using recent conversation context.

This is not full coreference resolution. It only substitutes a small set of
pronouns and definite noun phrases ('it', 'that project', 'the app') with the
most recent compatible entity from the current conversation.
"""

from __future__ import annotations

import re
from typing import Any

PRONOUNS = {
    r"\bthat project\b": "project",
    r"\bthis project\b": "project",
    r"\bthe project\b": "project",
    r"\bthe same project\b": "project",
    r"\bthe app\b": "project",
    r"\bthe application\b": "project",
    r"\bthe chatbot\b": "project",
    r"\bit\b": "any",
    r"\bthat\b": "any",
}


def _recent_project(context_messages: list[str], memories: list[dict[str, Any]]) -> str | None:
    for mem in memories:
        hint = (mem.get("subject") or "") + " " + " ".join(
            e.get("text", "") for e in mem.get("entities") or [] if e.get("label") == "PROJECT"
        )
        if "project" in hint.lower() or mem.get("attribute") == "language":
            subject = mem.get("subject")
            if subject and subject not in {"user", "user_project"}:
                return subject
    for text in reversed(context_messages):
        match = re.search(
            r"(project\s+[a-z0-9_\-]+|(?:college\s+)?project|chatbot|recallbot)",
            text.lower(),
        )
        if match:
            return match.group(1)
    return None


def resolve_coreferences(
    text: str,
    context_messages: list[str] | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context_messages = context_messages or []
    memories = memories or []
    resolved = text
    replacements: list[str] = []
    project = _recent_project(context_messages, memories)
    if not project:
        return {
            "text": text,
            "resolved": False,
            "replacements": [],
            "method": "lightweight_context_fallback",
            "note": (
                "No recent project/entity was available. Pronouns were left unchanged. "
                "This fallback is not full coreference resolution."
            ),
        }

    for pattern, kind in PRONOUNS.items():
        if kind in {"project", "any"} and re.search(pattern, resolved, flags=re.IGNORECASE):
            resolved = re.sub(pattern, project, resolved, count=1, flags=re.IGNORECASE)
            replacements.append(f"{pattern} -> {project}")
            break

    return {
        "text": resolved,
        "resolved": resolved != text,
        "replacements": replacements,
        "method": "lightweight_context_fallback",
        "note": (
            "Simple pronoun/NP substitution using the most recent project mention. "
            "This is not full coreference resolution."
        ),
    }
