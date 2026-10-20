"""Direct SQLite-backed fact recall, independent of vector similarity thresholds."""
from __future__ import annotations
import re
from typing import Any

from nlp.entity_extraction import extract_facts


def question_attribute(text: str) -> str | None:
    q = re.sub(r"\s+", " ", text.lower().strip())
    if re.search(r"\bwhat (?:do you )?remember about me\b", q):
        return "summary"
    if re.search(r"\b(?:what(?:'s| is)|tell me|remember|recall|know|do you remember|do you know).{0,35}\bmy name\b", q):
        return "name"
    if re.search(r"\b(?:where (?:do i live|am i (?:from|living))|which city (?:am i from|do i live in)|what(?:'s| is) my (?:city|location))\b", q):
        return "location"
    if re.search(r"\b(?:my (?:favorite|favourite) color|what color do i like)\b", q):
        return "favorite_color"
    if re.search(r"\b(?:my (?:favorite|favourite) food)\b", q):
        return "favorite_food"
    if re.search(r"\b(?:what (?:do i study|am i studying|is my (?:degree|major|field of study))|my degree subject)\b", q):
        return "degree_subject"
    if re.search(r"\b(?:how old am i|what(?:'s| is) my age)\b", q):
        return "age"
    if re.search(r"\bdo i like\s+([a-z0-9+#.-]+)", q):
        return "likes"
    return None


def _facts(mem: dict[str, Any]):
    # Prefer the memory's persisted atomic fields. Historical compound memories
    # might only have saved one field; extract missing fields from their text.
    if mem.get("attribute") and mem.get("value") is not None:
        yield {
            "subject": mem.get("subject"),
            "attribute": mem.get("attribute"),
            "value": mem.get("value"),
            "polarity": mem.get("polarity"),
        }
    for fact in extract_facts(mem.get("fact_text") or ""):
        parsed = fact.to_dict()
        if parsed.get("attribute") != "statement":
            yield parsed


def _answer(attr: str, value: str, polarity: str | None, question: str) -> str:
    if attr == "name":
        return f"Your name is {value}."
    if attr == "location":
        return f"You live in {value.title()}."
    if attr.startswith("favorite_"):
        kind = attr[len("favorite_"):].replace("_", " ")
        return f"Your favorite {kind} is {value}."
    if attr == "degree_subject":
        return f"You previously said you study {value}."
    if attr == "age":
        return f"You told me you are {value} years old."
    if attr == "likes":
        negative = polarity in {"negative", "negated"}
        return f"You told me you {'do not like' if negative else 'like'} {value}."
    return f"I remember: {value}."


def recall_direct(sqlite: Any, question: str) -> tuple[dict[str, Any], str] | None:
    attr = question_attribute(question)
    if attr is None:
        return None
    memories = sorted(sqlite.active_memories(), key=lambda m: int(m.get("is_demo") or 0))
    # Active memories are usually newest first. Structured matches win over
    # legacy text reconstruction, so old compound text does not mask updates.
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, mem in enumerate(memories):
        for fact in _facts(mem):
            if (fact.get("subject") or "").lower() != "user":
                continue
            if not fact.get("value"):
                continue
            if attr != "summary" and fact.get("attribute") != attr:
                continue
            if attr == "likes":
                requested = re.search(r"\bdo i like\s+([a-z0-9+#.-]+)", question.lower())
                if not requested or requested.group(1) != str(fact["value"]).lower():
                    continue
            candidates.append((index, mem, fact))
    if not candidates:
        return None
    # Group by attribute; use the newest active fact for each attribute.
    candidates.sort(key=lambda item: (item[0], 0 if item[1].get('attribute') == item[2].get('attribute') else 1))
    if attr == "summary":
        notes: dict[str, str] = {}
        for _idx, _mem, fact in candidates:
            a = str(fact.get("attribute") or "")
            if a not in notes and a in {"name", "location", "degree_subject", "favorite_color", "favorite_food", "age"}:
                notes[a] = str(fact["value"])
        if not notes:
            return None
        titles = {"name": "name", "location": "city", "degree_subject": "subject", "favorite_color": "favorite color", "favorite_food": "favorite food", "age": "age"}
        return candidates[0][1], "I remember: " + "; ".join(f"{titles[a]}: {v}" for a, v in notes.items()) + "."
    _idx, mem, fact = candidates[0]
    return mem, _answer(attr, str(fact["value"]), fact.get("polarity"), question)


def atomic_fact_text(fact: dict[str, Any]) -> str:
    attr = str(fact.get("attribute") or "")
    value = str(fact.get("value") or "")
    if attr == "name":
        return f"My name is {value}."
    if attr == "location":
        return f"I live in {value}."
    if attr.startswith("favorite_"):
        return f"My favorite {attr[len('favorite_'):].replace('_', ' ')} is {value}."
    if attr == "likes":
        negative = str(fact.get("polarity") or "").lower() in {"negative", "negated"}
        return f"I {'do not like' if negative else 'like'} {value}."
    return str(fact.get("text") or value)
