"""Template/rule-based replies grounded in retrieved memories only."""

from __future__ import annotations

from typing import Any

from nlp.lexicon import QUESTION_CUES
from nlp.preprocessing import preprocess_message


def is_question(text: str) -> bool:
    lower = text.lower().strip()
    if lower.endswith("?"):
        return True
    return any(lower.startswith(cue) or f" {cue} " in f" {lower} " for cue in QUESTION_CUES)


def _format_when(memory: dict[str, Any] | None) -> str:
    if not memory:
        return ""
    created = (memory.get("created_at") or "")[:10]
    return f" (recorded {created})" if created else ""


class ResponseGenerator:
    def generate(
        self,
        user_text: str,
        detection: dict[str, Any],
        retrieved: list[dict[str, Any]],
        coref: dict[str, Any] | None = None,
        nli_warning: str | None = None,
    ) -> dict[str, Any]:
        relation = detection.get("relation", "neutral")
        memory = detection.get("similar_memory")
        explanation = detection.get("explanation", "")
        question = is_question(user_text)

        if question:
            text = self._answer_question(user_text, retrieved)
        elif relation == "correction":
            old = memory.get("fact_text") if memory else "the earlier statement"
            text = (
                f"Understood — I'll treat this as a correction. I previously stored \"{old}\". "
                "That record stays in history as superseded, and the new statement is now active."
            )
        elif relation == "change_over_time":
            old = memory.get("fact_text") if memory else "an earlier detail"
            text = (
                f"This sounds like a change over time rather than an error. I still remember: \"{old}\". "
                "I can mark that memory as superseded if you confirm this is an update to the same thing. "
                "Did you migrate or change it, or is this a different item?"
            )
        elif relation == "contradiction":
            old = memory.get("fact_text") if memory else "an earlier statement"
            value_old = memory.get("value") if memory else "the previous value"
            value_new = (detection.get("new_fact") or {}).get("value")
            attr = (detection.get("new_fact") or {}).get("attribute") or "this detail"
            text = (
                f"I remember you previously mentioned {value_old} for {attr}, but now you've mentioned {value_new}. "
                f"Earlier I stored: \"{old}\". Did you switch, is this a correction, or are you referring to something else?"
            )
        elif relation == "ambiguous":
            old = memory.get("fact_text") if memory else "a related memory"
            text = (
                f"I found a related memory — \"{old}\" — but I am not confident it refers to the same thing. "
                "Could you clarify whether this is an update, a different project, or unrelated?"
            )
        elif relation == "entailment":
            old = memory.get("fact_text") if memory else "that"
            text = (
                f"That matches what I already have{ _format_when(memory)}: \"{old}\". "
                "I won't create a duplicate memory."
            )
        else:
            if retrieved and not question:
                top = retrieved[0]
                text = (
                    f"Got it. I'll remember that. "
                    f"A related earlier note was: \"{top.get('fact_text')}\"{_format_when(top)}."
                )
            else:
                text = "Got it. I'll remember that for later in this conversation history."

        if nli_warning:
            text += " (Advanced NLI is currently unavailable, so this comparison used conservative rules.)"

        if coref and coref.get("resolved"):
            text += f" I interpreted a reference using recent context as \"{coref.get('text')}\"."

        return {
            "text": text,
            "retrieved": retrieved,
            "relation": relation,
            "explanation": explanation,
        }

    def _answer_question(self, user_text: str, retrieved: list[dict[str, Any]]) -> str:
        prepared = preprocess_message(user_text)
        if not retrieved:
            return (
                "I couldn't find a sufficiently relevant memory about that. "
                "Could you tell me more?"
            )
        top = retrieved[0]
        when = _format_when(top)
        value = top.get("value")
        fact = top.get("fact_text")
        if "language" in prepared["lower"] or "java" in prepared["lower"] or "python" in prepared["lower"]:
            return (
                f"You previously mentioned {value or fact}{when}. "
                f"Stored wording: \"{fact}\". Is that the project you mean?"
            )
        if "degree" in prepared["lower"] or "stud" in prepared["lower"] or "subject" in prepared["lower"]:
            return f"From earlier{when}: \"{fact}\". I associated this with your studies."
        if "exam" in prepared["lower"]:
            return f"I have this exam-related memory{when}: \"{fact}\"."
        if "favorite" in prepared["lower"]:
            return f"You previously said{when}: \"{fact}\"."
        return (
            f"The most relevant memory I retrieved{when} is: \"{fact}\". "
            "I am only using stored user statements, not inventing extra details."
        )
