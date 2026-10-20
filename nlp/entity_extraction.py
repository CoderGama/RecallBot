"""NER plus rule-based extraction of project-specific facts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nlp.lexicon import (
    DAYS,
    DEGREE_TERMS,
    FRAMEWORKS,
    PROGRAMMING_LANGUAGES,
)
from nlp.preprocessing import preprocess_message

_SPACY_NLP = None
_SPACY_TRIED = False


def get_spacy():
    """Load spaCy lazily. Returns None if the model is unavailable."""
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy

        from config import SPACY_MODEL_NAME

        _SPACY_NLP = spacy.load(SPACY_MODEL_NAME)
    except Exception:
        try:
            import spacy

            _SPACY_NLP = spacy.blank("en")
            if "sentencizer" not in _SPACY_NLP.pipe_names:
                _SPACY_NLP.add_pipe("sentencizer")
        except Exception:
            _SPACY_NLP = None
    return _SPACY_NLP


@dataclass
class ExtractedFact:
    text: str
    subject: str
    attribute: str
    value: str
    polarity: str = "affirmative"
    confidence: float = 0.7
    fact_type: str = "generic"
    entities: list[dict[str, str]] = field(default_factory=list)
    project_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "polarity": self.polarity,
            "confidence": self.confidence,
            "fact_type": self.fact_type,
            "entities": self.entities,
            "project_hint": self.project_hint,
        }


def _polarity(text: str) -> str:
    lower = text.lower()
    negated = bool(
        re.search(
            r"\b(not|never|no longer|haven't|hasn't|didn't|don't|doesn't|did not|do not|have not|has not)\b",
            lower,
        )
    )
    return "negated" if negated else "affirmative"


def _find_languages(text: str) -> list[str]:
    found = []
    lower = text.lower()
    for lang in sorted(PROGRAMMING_LANGUAGES, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(lang) + r"(?![a-z])"
        if re.search(pattern, lower):
            found.append(lang if lang != "golang" else "go")
    return found


def _find_frameworks(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for item in sorted(FRAMEWORKS, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(item) + r"(?![a-z])", lower):
            found.append(item)
    return found


def _project_hint(text: str) -> str | None:
    patterns = [
        r"(project\s+(?!(?:to|in|with|using|is|was|for|on|from|into|and)\b)[a-z0-9_\-]+)",
        r"(the\s+[a-z0-9_\-]+\s+(?:project|app|application|chatbot|system|website))",
        r"((?:my|our)\s+(?:college\s+)?project)",
        r"\bbuilt\s+(?:a|an|the)\s+([a-z0-9_\-]+)",
        r"\b(chatbot|recallbot|website|application|app)\b",
    ]
    lower = text.lower()
    for pat in patterns:
        match = re.search(pat, lower)
        if match:
            return match.group(1).strip()
    return None


def extract_entities(text: str) -> list[dict[str, str]]:
    prepared = preprocess_message(text)
    if not prepared["valid"]:
        return []
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, label: str) -> None:
        key = (value.lower(), label)
        if value and key not in seen:
            seen.add(key)
            entities.append({"text": value, "label": label})

    nlp = get_spacy()
    if nlp is not None and hasattr(nlp, "pipe_names") and "ner" in nlp.pipe_names:
        doc = nlp(prepared["text"])
        for ent in doc.ents:
            add(ent.text, ent.label_)

    for lang in _find_languages(prepared["text"]):
        add(lang, "LANGUAGE")
    for fw in _find_frameworks(prepared["text"]):
        add(fw, "TECH")
    for degree in DEGREE_TERMS:
        if degree in prepared["lower"]:
            add(degree, "FIELD_OF_STUDY")
    for day in DAYS:
        if re.search(rf"\b{day}\b", prepared["lower"]):
            add(day, "DATE")
    hint = _project_hint(prepared["text"])
    if hint:
        add(hint, "PROJECT")

    date_match = re.search(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|october\s+\d{1,2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?\s+\d{1,2})\b",
        prepared["lower"],
    )
    if date_match:
        add(date_match.group(1), "DATE")
    return entities

def extract_compound_personal_facts(text):
    import re

    facts = []

    name_match = re.search(
        r"\bmy name is\s+([a-zA-Z]+)",
        text,
        re.IGNORECASE
    )

    if name_match:
        facts.append({
            "subject": "user",
            "attribute": "name",
            "value": name_match.group(1).strip(),
            "confidence": 0.95,
            "fact_type": "name",
        })

    location_match = re.search(
        r"\bi live in\s+([a-zA-Z]+)",
        text,
        re.IGNORECASE
    )

    if location_match:
        facts.append({
            "subject": "user",
            "attribute": "location",
            "value": location_match.group(1).strip(),
            "confidence": 0.95,
            "fact_type": "location",
        })

    return facts

def extract_facts(text: str) -> list[ExtractedFact]:
    prepared = preprocess_message(text)
    if not prepared["valid"]:
        return []
    raw = prepared["text"]
    lower = prepared["lower"]
    entities = extract_entities(raw)
    polarity = _polarity(raw)
    languages = _find_languages(raw)
    frameworks = _find_frameworks(raw)
    project = _project_hint(raw)
    facts: list[ExtractedFact] = []

    fav = re.search(
        r"(?:my\s+)?favorite\s+(programming\s+language|language|framework|tool|color|food|subject)\s+is\s+([a-z0-9\+\.# \-]+)",
        lower,
    )
    if fav:
        attr = "favorite_" + fav.group(1).replace("programming language", "language").replace(" ", "_")
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute=attr,
                value=re.split(r"\s+and\s+(?:i|my)\b", fav.group(2), maxsplit=1)[0].strip(" ."),
                polarity=polarity,
                confidence=0.9,
                fact_type="preference",
                entities=entities,
                project_hint=project,
            )
        )

    study = re.search(
        r"\b(?:study|studying|major(?:ing)? in|degree(?: subject)?(?: is)?|course is)\s+([a-z ]+)",
        lower,
    )
    if study or any(term in lower for term in DEGREE_TERMS if term != "it"):
        value = study.group(1).strip(" .") if study else next(
            (t for t in DEGREE_TERMS if t in lower and t != "it"), "computer science"
        )
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="degree_subject",
                value=value,
                polarity=polarity,
                confidence=0.86,
                fact_type="education",
                entities=entities,
            )
        )

    learn = re.search(r"\b(?:learning|learned|practising|practicing)\s+([a-z0-9\+\.#]+)", lower)
    if learn:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="learning",
                value=learn.group(1),
                polarity=polarity,
                confidence=0.8,
                fact_type="skill",
                entities=entities,
            )
        )

    lang_use = re.search(
        r"(?:developed|built|created|wrote|coded|implemented|made)\b.{0,80}?\b(?:using|in|with|entirely using)\s+([a-z0-9\+\.#]+)",
        lower,
    )
    mentions_using = bool(re.search(r"\b(using|uses|used|use|migrated|switched|rewrote|replaced)\b", lower))
    if languages and (lang_use or mentions_using) and not learn:
        value = languages[0]
        if lang_use:
            maybe = lang_use.group(1)
            if maybe in PROGRAMMING_LANGUAGES:
                value = maybe
        facts.append(
            ExtractedFact(
                text=raw,
                subject=project or "user_project",
                attribute="language",
                value=value,
                polarity=polarity,
                confidence=0.85,
                fact_type="project_language",
                entities=entities,
                project_hint=project,
            )
        )
    elif languages and not facts:
        facts.append(
            ExtractedFact(
                text=raw,
                subject=project or "user",
                attribute="language",
                value=languages[0],
                polarity=polarity,
                confidence=0.55,
                fact_type="language_mention",
                entities=entities,
                project_hint=project,
            )
        )

    project_named = re.search(r"\bproject\s+([a-z0-9_\-]+)\s+uses\s+([a-z0-9\+\.#]+)", lower)
    if project_named:
        facts.append(
            ExtractedFact(
                text=raw,
                subject=f"project {project_named.group(1)}",
                attribute="language",
                value=project_named.group(2),
                polarity=polarity,
                confidence=0.9,
                fact_type="project_language",
                entities=entities,
                project_hint=f"project {project_named.group(1)}",
            )
        )

    if frameworks:
        facts.append(
            ExtractedFact(
                text=raw,
                subject=project or "user",
                attribute="technology",
                value=frameworks[0],
                polarity=polarity,
                confidence=0.7,
                fact_type="technology",
                entities=entities,
                project_hint=project,
            )
        )

    exam = re.search(r"\bexam(?:\s+is)?(?:\s+on)?\s+([a-z0-9 ,/-]+)", lower)
    if exam:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="exam_date",
                value=exam.group(1).strip(" ."),
                polarity=polarity,
                confidence=0.88,
                fact_type="schedule",
                entities=entities,
            )
        )

    completed = re.search(r"\b(completed|finished|done with)\b.{0,40}\b(project|assignment|exam)?", lower)
    if completed:
        facts.append(
            ExtractedFact(
                text=raw,
                subject=project or "user_project",
                attribute="completion_status",
                value="completed" if polarity == "affirmative" else "not_completed",
                polarity=polarity,
                confidence=0.82,
                fact_type="status",
                entities=entities,
                project_hint=project,
            )
        )

    like = re.search(r"\bi (?:do not |don't |did not )?like\s+([a-z0-9\+\.#]+)", lower)
    if like:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="likes",
                value=like.group(1),
                polarity=polarity,
                confidence=0.78,
                fact_type="preference",
                entities=entities,
            )
        )

    age = re.search(r"\b(?:i am|i'm)\s+(\d{1,2})\s+years?\s+old", lower)
    if age:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="age",
                value=age.group(1),
                polarity="affirmative",
                confidence=0.9,
                fact_type="identity",
                entities=entities,
            )
        )

    live = re.search(r"\b(?:i live in|i'm from|i am from|my city is)\s+([a-z ]+)", lower)
    if live:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="location",
                value=re.split(r"\s+and\s+(?:i|my)\b", live.group(1), maxsplit=1)[0].strip(" ."),
                polarity=polarity,
                confidence=0.8,
                fact_type="location",
                entities=entities,
            )
        )

    name = re.search(r"\b(?:my name is|i am called|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", raw)
    if name and name.group(1).lower() not in {"studying", "learning", "using", "building"}:
        facts.append(
            ExtractedFact(
                text=raw,
                subject="user",
                attribute="name",
                value=name.group(1),
                polarity="affirmative",
                confidence=0.9,
                fact_type="identity",
                entities=entities,
            )
        )

    if not facts:
        facts.append(
            ExtractedFact(
                text=raw,
                subject=project or "user",
                attribute="statement",
                value=raw,
                polarity=polarity,
                confidence=0.4,
                fact_type="utterance",
                entities=entities,
                project_hint=project,
            )
        )

    # De-duplicate by attribute+value
    unique: list[ExtractedFact] = []
    seen = set()
    for fact in facts:
        key = (fact.attribute, fact.value, fact.subject)
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique
from copy import copy

# Preserve the existing extractor
_original_extract_facts = extract_facts


def extract_facts(text):
    # Run the original extraction logic
    facts = list(_original_extract_facts(text))

    # Extract additional personal facts
    personal_facts = extract_compound_personal_facts(text)

    for item in personal_facts:

        # Avoid adding the same fact twice
        already_exists = any(
            f.subject == item["subject"]
            and f.attribute == item["attribute"]
            and str(f.value).strip().rstrip(".!?").casefold()
            == str(item["value"]).strip().rstrip(".!?").casefold()
            for f in facts
        )

        if already_exists:
            continue

        # Reuse the existing fact object's structure
        if not facts:
            continue

        new_fact = copy(facts[0])

        new_fact.text = text
        new_fact.subject = item["subject"]
        new_fact.attribute = item["attribute"]
        new_fact.value = item["value"]
        new_fact.polarity = "affirmative"
        new_fact.confidence = item["confidence"]
        new_fact.fact_type = item["fact_type"]
        new_fact.entities = []
        new_fact.project_hint = None

        facts.append(new_fact)

    return facts