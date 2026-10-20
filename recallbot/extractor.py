"""Rule-based fact extraction enhanced by spaCy tokenization and sentence splitting.

Explicit, high-precision patterns are preferable to inventing personal facts.
spaCy NER is available for inspection when en_core_web_sm is installed.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .models import Fact


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(" .!?;,:\"'" )).casefold()


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(" .!?;,:\"'")


def _add(results: list[Fact], attribute: str, value: str, sentence: str):
    value = _tidy(value)
    if not value or len(value) > 100 or value.lower() in {"a", "an", "the", "it"}:
        return
    fact = Fact(attribute=attribute, value=value, statement=sentence)
    if not any(f.attribute == fact.attribute and normalize(f.value) == normalize(fact.value) for f in results):
        results.append(fact)


# These attributes represent one current value. Changing one requires clarification.
LABELS = {
    "name": "Name",
    "lives_in": "Current city/location",
    "from": "Hometown/origin",
    "works_at": "Workplace",
    "studies_at": "College/school",
    "studies": "Field of study",
    "age": "Age",
    "occupation": "Occupation",
    "favorite_color": "Favorite color",
    "favorite_food": "Favorite food",
    "favorite_language": "Favorite programming language",
    "favorite_game": "Favorite game",
    "pet_name": "Pet's name",
}


def label_for(attribute: str) -> str:
    if attribute.startswith("likes:"):
        return "Likes " + attribute.split(":", 1)[1]
    return LABELS.get(attribute, attribute.replace("_", " ").capitalize())


def extract_facts(text: str) -> list[Fact]:
    text = text.strip()
    if not text or "?" in text:
        return []
    doc = get_nlp()(text)
    sentences = [s.text.strip() for s in doc.sents] if doc.has_annotation("SENT_START") else [text]
    facts: list[Fact] = []
    for sentence in sentences:
        # Multiple first-person claims in a sentence: "My name is A and I live in B".
        segments = re.split(r"\s+(?:and|but)\s+(?=(?:I|my)\b)", sentence, flags=re.I)
        for seg in segments:
            s = _tidy(seg)
            if not s:
                continue
            patterns = [
                (r"^(?:my name is|i am called|call me)\s+(.+)$", "name"),
                (r"^(?:i live in|i am living in|i currently live in|i moved to|my current city is|my city is)\s+(.+)$", "lives_in"),
                (r"^(?:i am from|i come from|my hometown is)\s+(.+)$", "from"),
                (r"^(?:i work at|i work for|my employer is|my company is)\s+(.+)$", "works_at"),
                (r"^(?:i study at|i go to college at|my college is|my university is|my school is)\s+(.+)$", "studies_at"),
                (r"^(?:i study|i am studying|my major is|my field of study is)\s+(.+)$", "studies"),
                (r"^(?:i am|i'm)\s+(\d{1,3})\s+years? old$", "age"),
                (r"^(?:my age is)\s+(\d{1,3})$", "age"),
                (r"^(?:i am an?|i'm an?)\s+(student|developer|engineer|designer|teacher|researcher|intern|programmer)$", "occupation"),
                (r"^my (?:favorite|favourite) (?:colour|color) is\s+(.+)$", "favorite_color"),
                (r"^my (?:favorite|favourite) food is\s+(.+)$", "favorite_food"),
                (r"^my (?:favorite|favourite) (?:programming )?language is\s+(.+)$", "favorite_language"),
                (r"^my (?:favorite|favourite) game is\s+(.+)$", "favorite_game"),
                (r"^my pet(?:'s|s) name is\s+(.+)$", "pet_name"),
            ]
            matched = False
            for pattern, key in patterns:
                m = re.fullmatch(pattern, s, flags=re.I)
                if m:
                    _add(facts, key, m.group(1), s)
                    matched = True
                    break
            if matched:
                continue
            # General personal property, e.g. "My project is RecallBot".
            m = re.fullmatch(r"my ([a-z][a-z ]{1,35}) is (.+)", s, flags=re.I)
            if m:
                key = "custom:" + re.sub(r"\s+", "_", m.group(1).strip().casefold())
                _add(facts, key, m.group(2), s)
                continue
            m = re.fullmatch(r"i (do not like|don't like|really don't like|dislike|like|love|enjoy)\s+(.+)", s, flags=re.I)
            if m:
                expression = m.group(1).casefold()
                object_text = _tidy(m.group(2))
                # No conflation of liking football and liking cricket.
                key = "likes:" + normalize(object_text)
                value = "no" if expression in {"do not like", "don't like", "really don't like", "dislike"} else "yes"
                _add(facts, key, value, s)
    return facts


def attribute_from_question(text: str) -> str | None:
    q = normalize(text)
    # First-person queries should be interpreted as queries about the speaker.
    if re.search(r"\b(?:my name|who am i|what do you call me)\b", q):
        return "name"
    if re.search(r"\b(?:where do i live|where am i living|my current city|my location|which city do i live)\b", q):
        return "lives_in"
    if re.search(r"\b(?:where am i from|my hometown)\b", q):
        return "from"
    if re.search(r"\b(?:where do i work|who do i work for|my workplace|my employer)\b", q):
        return "works_at"
    if re.search(r"\b(?:where do i study|my college|my university|my school)\b", q):
        return "studies_at"
    if re.search(r"\b(?:what do i study|my major|my field of study)\b", q):
        return "studies"
    if re.search(r"\b(?:how old am i|what is my age|what's my age)\b", q):
        return "age"
    if re.search(r"\b(?:my occupation|what is my job|what do i do for a living)\b", q):
        return "occupation"
    for thing, attr in [("color", "favorite_color"), ("colour", "favorite_color"), ("food", "favorite_food"), ("language", "favorite_language"), ("game", "favorite_game")]:
        if re.search(r"\b(?:favorite|favourite)\b.*\b" + thing + r"\b", q):
            return attr
    m = re.search(r"\b(?:do i like|do i enjoy|do i love)\s+(.+)", q)
    if m:
        return "likes:" + normalize(m.group(1))
    m = re.search(r"\bwhat is my ([a-z ]+)\b", q)
    if m:
        return "custom:" + re.sub(r"\s+", "_", m.group(1).strip())
    return None
