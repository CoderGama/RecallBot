"""Combine retrieval, structured facts, rules, and NLI to estimate contradictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import NLI_CONFIDENCE_THRESHOLD, STRONG_SIMILARITY_THRESHOLD
from nlp.lexicon import CHANGE_MARKERS, CORRECTION_MARKERS
from nlp.nli_classifier import NLIClassifier, NLIResult

COMPATIBLE_ATTR_PAIRS = {
    ("degree_subject", "learning"),
    ("learning", "degree_subject"),
    ("technology", "language"),
    ("language", "technology"),
}


def _is_correction(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CORRECTION_MARKERS)


def _is_change(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CHANGE_MARKERS)


def _projects_differ(a: dict[str, Any], b: dict[str, Any]) -> bool:
    pa = (a.get("project_hint") or a.get("subject") or "").lower().strip()
    pb = (b.get("project_hint") or b.get("subject") or "").lower().strip()
    generic = {"", "user", "user_project", "project", "the project", "my project", "our project"}
    if pa in generic or pb in generic:
        return False
    if "project" in pa and "project" in pb:
        return pa != pb
    return False


def _entity_overlap(mem_entities: list[dict[str, str]], fact_entities: list[dict[str, str]]) -> bool:
    a = {(e.get("text", "").lower(), e.get("label")) for e in mem_entities or []}
    b = {(e.get("text", "").lower(), e.get("label")) for e in fact_entities or []}
    return bool(a & b)


@dataclass
class DetectionResult:
    relation: str
    confidence: float
    should_ask: bool
    explanation: str
    similar_memory: dict[str, Any] | None = None
    new_fact: dict[str, Any] | None = None
    nli: NLIResult | None = None
    similarity: float = 0.0
    policy: str = "store_if_new"
    technical: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "confidence": self.confidence,
            "should_ask": self.should_ask,
            "explanation": self.explanation,
            "similar_memory": self.similar_memory,
            "new_fact": self.new_fact,
            "nli": self.nli.to_dict() if self.nli else None,
            "similarity": self.similarity,
            "policy": self.policy,
            "technical": self.technical,
        }


class ContradictionDetector:
    def __init__(self, nli: NLIClassifier | None = None) -> None:
        self.nli = nli or NLIClassifier()

    def detect(
        self,
        message: str,
        facts: list[Any],
        retrieved: list[dict[str, Any]],
    ) -> DetectionResult:
        fact_dicts = [f.to_dict() if hasattr(f, "to_dict") else f for f in facts]
        primary = fact_dicts[0] if fact_dicts else {
            "text": message,
            "subject": "user",
            "attribute": "statement",
            "value": message,
            "polarity": "affirmative",
            "entities": [],
            "project_hint": None,
        }

        if not retrieved:
            return DetectionResult(
                relation="neutral",
                confidence=0.7,
                should_ask=False,
                explanation="No sufficiently similar memory was retrieved for comparison.",
                new_fact=primary,
                policy="store_if_new",
                technical={"retrieved": 0},
            )

        if _is_correction(message):
            best = retrieved[0]
            return DetectionResult(
                relation="correction",
                confidence=0.86,
                should_ask=False,
                explanation=(
                    "The new message contains an explicit correction marker. "
                    "The earlier memory will be marked superseded and kept in history."
                ),
                similar_memory=best,
                new_fact=primary,
                similarity=float(best.get("similarity") or 0.0),
                policy="supersede_as_correction",
                technical={"reason": "explicit_correction"},
            )

        if _is_change(message):
            best = retrieved[0]
            return DetectionResult(
                relation="change_over_time",
                confidence=0.78,
                should_ask=True,
                explanation=(
                    "The wording suggests a change over time (for example migrated/switched). "
                    "This may update a preference or project language rather than conflict with the past."
                ),
                similar_memory=best,
                new_fact=primary,
                similarity=float(best.get("similarity") or 0.0),
                policy="supersede_as_change",
                technical={"reason": "change_marker"},
            )

        best_conflict: DetectionResult | None = None
        for memory in retrieved:
            candidate = self._compare(message, primary, fact_dicts, memory)
            if best_conflict is None:
                best_conflict = candidate
            else:
                rank = {
                    "contradiction": 4,
                    "ambiguous": 3,
                    "correction": 3,
                    "change_over_time": 2,
                    "entailment": 1,
                    "neutral": 0,
                }
                if rank.get(candidate.relation, 0) > rank.get(best_conflict.relation, 0):
                    best_conflict = candidate
                elif (
                    candidate.relation == best_conflict.relation
                    and candidate.confidence > best_conflict.confidence
                ):
                    best_conflict = candidate
        return best_conflict or DetectionResult(
            relation="neutral",
            confidence=0.5,
            should_ask=False,
            explanation="Retrieved memories did not yield a confident relationship.",
            new_fact=primary,
            policy="store_if_new",
        )

    def _compare(
        self,
        message: str,
        primary: dict[str, Any],
        facts: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> DetectionResult:
        similarity = float(memory.get("similarity") or 0.0)
        mem_attr = (memory.get("attribute") or "").lower()
        fact = next((f for f in facts if f.get("attribute") == mem_attr), primary)
        attr = (fact.get("attribute") or "").lower()
        different_projects = any(_projects_differ(f, memory) for f in facts) or _projects_differ(fact, memory)
        overlap = _entity_overlap(memory.get("entities") or [], fact.get("entities") or [])
        same_attr = bool(attr and mem_attr and attr == mem_attr and attr != "statement")
        different_values = same_attr and (str(fact.get("value") or "").lower() != str(memory.get("value") or "").lower())
        polarity_flip = (
            same_attr
            and fact.get("polarity")
            and memory.get("polarity")
            and fact.get("polarity") != memory.get("polarity")
        )
        compatible_pair = any(
            (str(f.get("attribute") or "").lower(), mem_attr) in COMPATIBLE_ATTR_PAIRS
            or (mem_attr, str(f.get("attribute") or "").lower()) in COMPATIBLE_ATTR_PAIRS
            for f in facts
        )

        nli = self.nli.predict(memory.get("fact_text") or "", message)
        nli_contradiction = (
            nli.label == "contradiction" and nli.confidence >= NLI_CONFIDENCE_THRESHOLD
        )
        nli_entailment = nli.label == "entailment" and nli.confidence >= NLI_CONFIDENCE_THRESHOLD

        technical = {
            "same_attribute": same_attr,
            "different_values": different_values,
            "polarity_flip": polarity_flip,
            "different_projects": different_projects,
            "entity_overlap": overlap,
            "similarity": similarity,
            "nli": nli.to_dict(),
            "compatible_pair": compatible_pair,
        }

        if different_projects:
            return DetectionResult(
                relation="neutral",
                confidence=0.82,
                should_ask=False,
                explanation=(
                    "The statements appear to refer to different projects or subjects, "
                    "so different details can coexist."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="store_if_new",
                technical=technical,
            )

        if compatible_pair:
            return DetectionResult(
                relation="neutral",
                confidence=0.8,
                should_ask=False,
                explanation=(
                    "The statements are compatible: they describe related but different "
                    "attributes (for example a degree subject and a language being learned)."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="store_if_new",
                technical=technical,
            )

        if polarity_flip and (same_attr or (attr == "statement" and similarity >= STRONG_SIMILARITY_THRESHOLD)):
            return DetectionResult(
                relation="contradiction",
                confidence=0.84,
                should_ask=True,
                explanation=(
                    "The new statement negates an earlier claim about the same attribute. "
                    "This is treated as a potential contradiction, not as proof."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="hold_for_review",
                technical=technical,
            )

        if same_attr and different_values:
            conf = 0.8 if similarity >= STRONG_SIMILARITY_THRESHOLD else 0.68
            return DetectionResult(
                relation="contradiction",
                confidence=conf,
                should_ask=True,
                explanation=(
                    f"Both statements describe '{attr}' but the values differ "
                    f"({memory.get('value')} vs {fact.get('value')}). "
                    "This may be a changed preference, a correction, or a different referent."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="hold_for_review",
                technical=technical,
            )

        if nli_contradiction and similarity >= STRONG_SIMILARITY_THRESHOLD:
            return DetectionResult(
                relation="contradiction",
                confidence=min(0.75, 0.4 + nli.confidence * 0.4 + similarity * 0.2),
                should_ask=True,
                explanation=(
                    "NLI estimates a contradiction and the memories are semantically close. "
                    "The estimate is uncertain enough that clarification is requested."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="hold_for_review",
                technical=technical,
            )

        if nli_contradiction and similarity < STRONG_SIMILARITY_THRESHOLD:
            return DetectionResult(
                relation="ambiguous",
                confidence=0.55,
                should_ask=True,
                explanation=(
                    "NLI leaned toward contradiction, but similarity and entity overlap "
                    "are not strong enough to treat it as the same fact. Clarification is safer."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="hold_for_review",
                technical=technical,
            )

        if nli_entailment or (same_attr and not different_values):
            return DetectionResult(
                relation="entailment",
                confidence=max(nli.confidence, 0.6),
                should_ask=False,
                explanation="The new statement appears to support or repeat a stored memory.",
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="skip_duplicate",
                technical=technical,
            )

        if similarity >= STRONG_SIMILARITY_THRESHOLD and not overlap and attr == "statement":
            return DetectionResult(
                relation="ambiguous",
                confidence=0.5,
                should_ask=True,
                explanation=(
                    "A related memory was retrieved, but it is unclear whether both messages "
                    "refer to the same thing."
                ),
                similar_memory=memory,
                new_fact=fact,
                nli=nli,
                similarity=similarity,
                policy="hold_for_review",
                technical=technical,
            )

        return DetectionResult(
            relation="neutral",
            confidence=0.66,
            should_ask=False,
            explanation=(
                "Retrieved text is related enough to recall, but there is not enough evidence "
                "of a genuine contradiction."
            ),
            similar_memory=memory,
            new_fact=fact,
            nli=nli,
            similarity=similarity,
            policy="store_if_new",
            technical=technical,
        )
