"""Orchestrate preprocessing, retrieval, contradiction checks, and replies."""

from __future__ import annotations

import re

from typing import Any

from chatbot.memory_manager import MemoryManager

from chatbot.response_generator import ResponseGenerator, is_question

from config import MEMORY_STATUS_ACTIVE, MEMORY_STATUS_REVIEW

from database.sqlite_manager import SQLiteManager

from nlp.contradiction_detector import ContradictionDetector

from nlp.coreference import resolve_coreferences

from nlp.entity_extraction import extract_entities, extract_facts

from nlp.preprocessing import preprocess_message

from nlp.semantic_retrieval import SemanticRetriever

from nlp.direct_recall import recall_direct

# Only explicit names are treated as distinct project identities.

# E.g. "Project Alpha" is named, but "the project to Python" is not.

_NAMED_PROJECT = re.compile(

    r"\bproject\s+(?!to\b|in\b|with\b|using\b|is\b|was\b|for\b|on\b|from\b|into\b|and\b|a\b|the\b|this\b|my\b|our\b)([a-z][a-z0-9_-]\*)\b",

    flags=re.IGNORECASE,

)

_PROJECT_ATTRIBUTES = {"language", "technology", "completion_status"}

def _named_project(text: str | None) -> str | None:

    """Return a canonical named-project subject, or None for generic references."""

    match = _NAMED_PROJECT.search(str(text or ""))

    return f"project {match.group(1).lower()}" if match else None

def _memory_named_project(memory: dict[str, Any]) -> str | None:

    """Prefer the memory subject, and fall back to stored project metadata."""

    metadata = memory.get("metadata") or {}

    if not isinstance(metadata, dict):

        metadata = {}

    return (

        _named_project(memory.get("subject"))

        or _named_project(metadata.get("project_hint"))

        or _named_project(memory.get("project_hint"))

    )

class ConversationManager:

    def __init__(

        self,

        sqlite: SQLiteManager,

        memory: MemoryManager,

        retriever: SemanticRetriever,

        detector: ContradictionDetector,

        responder: ResponseGenerator | None = None,

    ) -> None:

        self.sqlite = sqlite

        self.memory = memory

        self.retriever = retriever

        self.detector = detector

        self.responder = responder or ResponseGenerator()

    def process(

        self,

        conversation_id: str,

        user_text: str,

        is_demo: bool = False,

    ) -> dict[str, Any]:

        prepared = preprocess_message(user_text)

        if not prepared["valid"]:

            return {

                "ok": False,

                "assistant_text": "Please enter a short, non-empty message that I can process.",

                "error": "invalid_input",

            }

        prior = self.sqlite.list_messages(conversation_id)

        context = [m["content"] for m in prior if m["role"] == "user"][-6:]

        active = self.sqlite.active_memories()

        coref = resolve_coreferences(prepared["text"], context, active)

        working_text = coref["text"]

        user_msg_id = self.sqlite.add_message(

            conversation_id,

            "user",

            prepared["text"],

            metadata={"coref": coref},

        )

        entities = extract_entities(working_text)

        self.sqlite.insert_entities(entities, user_msg_id)

        facts = extract_facts(working_text)

        # Explicit names in the user's ORIGINAL input take priority over

        # lightweight coreference guesses. Separate named projects must not

        # share a memory identity (Project Alpha != Project Beta).

        named_project = _named_project(prepared["text"])

        if named_project and not is_question(working_text):

            for fact in facts:

                if fact.attribute in _PROJECT_ATTRIBUTES:

                    fact.subject = named_project

                    fact.project_hint = named_project

        retrieved = self.retriever.search(

            working_text,

            sqlite_lookup=self.sqlite.get_memory,

        )

        if named_project and not is_question(working_text):

            project_only = bool(facts) and all(

                f.attribute in _PROJECT_ATTRIBUTES for f in facts

            )

            # The vector search can return related sentences about entirely

            # different named projects. Drop those candidates before NLI.

            # For project-only messages, also drop generic memories: there is

            # no evidence they refer to the same named project.

            retrieved = [

                memory

                for memory in retrieved

                if (

                    _memory_named_project(memory) == named_project

                    if project_only

                    else _memory_named_project(memory) in {None, named_project}

                )

            ]

        direct_hit = recall_direct(self.sqlite, working_text) if is_question(working_text) else None

        if direct_hit:

            retrieved = [direct_hit[0]]

        self.sqlite.log_retrieval(

            working_text,

            [m["id"] for m in retrieved],

            [float(m.get("similarity") or 0.0) for m in retrieved],

            conversation_id,

        )

        detection = self.detector.detect(working_text, facts, retrieved)

        det = detection.to_dict()

        stored_ids: list[str] = []

        event_id = None

        if not is_question(working_text):

            stored_ids, event_id = self._apply_policy_all(

                conversation_id, user_msg_id, facts, det, is_demo

            )

        nli_warning = None

        if not self.detector.nli.available:

            nli_warning = self.detector.nli.warning

        reply = self.responder.generate(

            working_text,

            det,

            retrieved,

            coref=coref,

            nli_warning=nli_warning if det.get("relation") in {"contradiction", "ambiguous"} else None,

        )

        if direct_hit:

            reply["text"] = direct_hit[1]

        assistant_meta = {

            "detection": det,

            "retrieved_ids": [m["id"] for m in retrieved],

            "event_id": event_id,

            "stored_memory_ids": stored_ids,

            "coref": coref,

            "entities": entities,

            "nli_backend": self.detector.nli.backend,

            "embedding_backend": self.memory.embedder.backend,

        }

        assistant_id = self.sqlite.add_message(

            conversation_id,

            "assistant",

            reply["text"],

            metadata=assistant_meta,

        )

        return {

            "ok": True,

            "assistant_text": reply["text"],

            "user_message_id": user_msg_id,

            "assistant_message_id": assistant_id,

            "retrieved": retrieved,

            "detection": det,

            "entities": entities,

            "facts": [f.to_dict() for f in facts],

            "event_id": event_id,

            "stored_memory_ids": stored_ids,

            "coref": coref,

            "nli_backend": self.detector.nli.backend,

            "embedding_backend": self.memory.embedder.backend,

            "warnings": {

                "nli": self.detector.nli.warning,

                "embeddings": self.memory.embedder.warning,

            },

        }

    def _apply_policy(

        self,

        conversation_id: str,

        message_id: str,

        facts,

        det: dict[str, Any],

        is_demo: bool,

    ) -> tuple[list[str], str | None]:

        stored: list[str] = []

        event_id = None

        policy = det.get("policy")

        primary = det.get("new_fact") or (facts[0].to_dict() if facts else None)

        if not primary:

            return stored, None

        record = {

            "fact_text": primary.get("text"),

            "subject": primary.get("subject"),

            "attribute": primary.get("attribute"),

            "value": primary.get("value"),

            "source_message_id": message_id,

            "conversation_id": conversation_id,

            "confidence": primary.get("confidence", 0.7),

            "status": MEMORY_STATUS_ACTIVE,

            "is_demo": int(is_demo),

            "polarity": primary.get("polarity"),

            "entities": primary.get("entities") or [],

            "metadata": {"fact_type": primary.get("fact_type"), "project_hint": primary.get("project_hint")},

        }

        old = det.get("similar_memory")

        if policy == "skip_duplicate":

            return stored, None

        if policy == "supersede_as_correction" and old:

            stored.append(

                self.memory.supersede(old["id"], record, reason="explicit_correction")

            )

        elif policy == "supersede_as_change" and old:

            record["status"] = MEMORY_STATUS_REVIEW

            stored.append(self.memory.add_memory(record, skip_duplicate=False)["id"])

        elif policy == "hold_for_review":

            record["status"] = MEMORY_STATUS_REVIEW

            stored.append(self.memory.add_memory(record, skip_duplicate=False)["id"])

        else:

            result = self.memory.add_memory(record, skip_duplicate=True)

            if result.get("status") != "duplicate":

                stored.append(result["id"])

        if det.get("relation") in {"contradiction", "ambiguous", "change_over_time"} and old:

            event_id = self.sqlite.insert_contradiction(

                {

                    "new_statement": primary.get("text"),

                    "old_memory_id": old.get("id"),

                    "old_statement": old.get("fact_text"),

                    "predicted_relation": det.get("relation"),

                    "confidence": det.get("confidence"),

                    "similarity": det.get("similarity"),

                    "explanation": det.get("explanation"),

                    "status": "open",

                    "is_demo": int(is_demo),

                    "conversation_id": conversation_id,

                    "nli_available": int(self.detector.nli.available),

                    "technical": det.get("technical") or {},

                }

            )

        return stored, event_id

    def _apply_policy_all(

        self,

        conversation_id: str,

        message_id: str,

        facts,

        det: dict[str, Any],

        is_demo: bool,

    ) -> tuple[list[str], str | None]:

        """Persist each extracted fact, compare matching attributes, and review conflicts."""

        from nlp.direct_recall import atomic_fact_text

        from nlp.contradiction_detector import _is_change, _is_correction

        stored: list[str] = []

        first_event_id: str | None = None

        conflict_result = None

        correction_result = None

        change_result = None

        duplicate_memory = None

        active = self.sqlite.active_memories()

        new_count = 0

        used_original_policy = False

        # Extractor returns one fact for unknown utterances and multiple for

        # compound statements. Do not discard facts after index 0.

        for raw_fact in facts:

            fact = raw_fact.to_dict() if hasattr(raw_fact, "to_dict") else raw_fact

            attr = str(fact.get("attribute") or "statement").lower()

            subject = str(fact.get("subject") or "user").lower()

            value = str(fact.get("value") or "").strip().rstrip(".!?")

            polarity = str(fact.get("polarity") or "affirmative").lower()

            project = str(fact.get("project_hint") or "").lower()

            if not value:

                continue

            # Preserve the original semantic/NLI policy for unstructured utterances.

            if attr == "statement":

                used_original_policy = True

                original_ids, original_event = self._apply_policy(

                    conversation_id, message_id, [raw_fact], det, is_demo

                )

                stored.extend(original_ids)

                if first_event_id is None:

                    first_event_id = original_event

                if original_ids:

                    new_count += len(original_ids)

                continue

            relevant = [m for m in active if

                        str(m.get("attribute") or "").lower() == attr and

                        str(m.get("subject") or "").lower() == subject and

                        str((m.get("metadata") or {}).get("project_hint") or "").lower() == project and

                        int(m.get("is_demo") or 0) == int(is_demo)]

            same = next((m for m in relevant if

                         str(m.get("value") or "").strip().rstrip(".!?").casefold() == value.casefold() and

                         str(m.get("polarity") or "affirmative").lower() == polarity), None)

            if same is not None:

                duplicate_memory = same

                continue

            old = relevant[0] if relevant else None

            single_value = attr in {"name", "location", "age", "degree_subject", "exam_date", "completion_status"} or attr.startswith("favorite_") or (attr == "language" and subject != "user")

            polarity_conflict = (old is not None and attr == "likes" and

                                 str(old.get("value") or "").strip().rstrip(".!?").casefold() == value.casefold() and

                                 str(old.get("polarity") or "affirmative").lower() != polarity)

            conflicting = old is not None and ((single_value and

                            str(old.get("value") or "").strip().rstrip(".!?").casefold() != value.casefold())

                            or polarity_conflict)

            record = {

                "fact_text": atomic_fact_text(fact),

                "subject": fact.get("subject"),

                "attribute": fact.get("attribute"),

                "value": value,

                "source_message_id": message_id,

                "conversation_id": conversation_id,

                "confidence": fact.get("confidence", 0.7),

                "status": MEMORY_STATUS_REVIEW if conflicting else MEMORY_STATUS_ACTIVE,

                "is_demo": int(is_demo),

                "polarity": fact.get("polarity"),

                "entities": fact.get("entities") or [],

                "metadata": {"fact_type": fact.get("fact_type"),

                             "project_hint": fact.get("project_hint"),

                             "source_text": fact.get("text")},

            }

            # Explicit corrections supersede the old value automatically.

            if old is not None and _is_correction(str(fact.get("text") or "")):

                corrected = dict(record)

                corrected["status"] = MEMORY_STATUS_ACTIVE

                new_id = self.memory.supersede(

                    old["id"], corrected, reason="explicit_correction"

                )

                stored.append(new_id)

                new_count += 1

                correction_result = {

                    "relation": "correction", "confidence": 0.9,

                    "should_ask": False,

                    "explanation": "Explicit correction applied; the old fact was preserved as superseded.",

                    "similar_memory": old, "new_fact": fact,

                    "policy": "supersede_as_correction",

                    "technical": {"reason": "explicit_correction"},

                }

                if old in active:

                    active.remove(old)

                fresh = self.sqlite.get_memory(new_id)

                if fresh is not None:

                    active.insert(0, fresh)

                continue

            # A time-based change is different from a contradiction. Ask for

            # confirmation and keep the previous active fact until resolved.

            if old is not None and _is_change(str(fact.get("text") or "")):

                proposed = dict(record)

                proposed["status"] = MEMORY_STATUS_REVIEW

                new_id = self.memory.add_memory(proposed, skip_duplicate=False)["id"]

                stored.append(new_id)

                new_count += 1

                change_result = {

                    "relation": "change_over_time", "confidence": 0.85,

                    "should_ask": True,

                    "explanation": "A change over time was mentioned. Confirm before updating the active memory.",

                    "similar_memory": old, "new_fact": fact,

                    "policy": "supersede_as_change",

                    "technical": {"reason": "change_marker", "new_review_memory_id": new_id},

                }

                event = self.sqlite.insert_contradiction({

                    "new_statement": proposed["fact_text"],

                    "old_memory_id": old["id"],

                    "old_statement": old.get("fact_text"),

                    "predicted_relation": "change_over_time",

                    "confidence": 0.85,

                    "similarity": 1.0,

                    "explanation": change_result["explanation"],

                    "status": "open", "is_demo": int(is_demo),

                    "conversation_id": conversation_id,

                    "nli_available": int(self.detector.nli.available),

                    "technical": change_result["technical"],

                })

                if first_event_id is None:

                    first_event_id = event

                continue

            result = self.memory.add_memory(record, skip_duplicate=not conflicting)

            if result.get("status") == "duplicate":

                duplicate_memory = result.get("memory")

                continue

            new_id = result["id"]

            stored.append(new_id)

            new_count += 1

            if conflicting:

                conflict_result = {

                    "relation": "contradiction", "confidence": 0.9,

                    "should_ask": True,

                    "explanation": "The saved and new values differ for the same attribute.",

                    "similar_memory": old,

                    "new_fact": fact,

                    "similarity": 1.0,

                    "policy": "hold_for_review",

                    "technical": {"same_attribute": True,

                                  "different_values": True,

                                  "new_review_memory_id": new_id},

                }

                event = self.sqlite.insert_contradiction({

                    "new_statement": record["fact_text"],

                    "old_memory_id": old["id"],

                    "old_statement": old.get("fact_text"),

                    "predicted_relation": "contradiction",

                    "confidence": 0.9,

                    "similarity": 1.0,

                    "explanation": conflict_result["explanation"],

                    "status": "open", "is_demo": int(is_demo),

                    "conversation_id": conversation_id,

                    "nli_available": int(self.detector.nli.available),

                    "technical": conflict_result["technical"],

                })

                if first_event_id is None:

                    first_event_id = event

            else:

                # Incorporate newly saved memories for repeated facts in a single turn.

                new_mem = self.sqlite.get_memory(new_id)

                if new_mem is not None and new_mem.get("status") == MEMORY_STATUS_ACTIVE:

                    active.insert(0, new_mem)

        # Preserve explicit time/correction intent before generic value-clash policy.

        if correction_result is not None:

            det.update(correction_result)

        elif change_result is not None:

            det.update(change_result)

        elif conflict_result is not None:

            det.update(conflict_result)

        elif used_original_policy:

            pass  # The original NLI/semantic result should remain visible.

        elif new_count:

            det.update({"relation": "neutral", "policy": "store_if_new",

                        "should_ask": False, "similar_memory": None,

                        "explanation": f"Stored {new_count} atomic fact(s)."})

        elif duplicate_memory:

            det.update({"relation": "entailment", "policy": "skip_duplicate",

                        "should_ask": False, "similar_memory": duplicate_memory,

                        "explanation": "This fact is already in active memory."})

        if correction_result is None and change_result is None and conflict_result is None:

            import re

            old_memory = det.get("similar_memory") or {}

            old_subject = str(old_memory.get("subject") or "").lower().strip()

            new_subjects = {

                str((f.to_dict() if hasattr(f, "to_dict") else f).get("subject") or "").lower().strip()

                for f in facts

            }

            # Explicitly named projects only; avoid inferring different

            # projects from vague references like 'this project'.

            project_pattern = r"project [a-z0-9_-]+"

            if (re.fullmatch(project_pattern, old_subject)

                    and new_subjects

                    and all(re.fullmatch(project_pattern, s) and s != old_subject

                            for s in new_subjects)):

                det.update({

                    "relation": "neutral", "policy": "store_if_new",

                    "should_ask": False,

                    "explanation": "The facts refer to distinct named projects.",

                })

        return stored, first_event_id

    def resolve_event(self, event_id: str, action: str) -> dict[str, Any]:

        event = self.sqlite.get_contradiction(event_id)

        if not event:

            return {"ok": False, "error": "Event not found."}

        old_id = event.get("old_memory_id")

        new_matches = [

            m

            for m in self.sqlite.list_memories(status=MEMORY_STATUS_REVIEW)

            if m.get("fact_text") == event.get("new_statement")

        ]

        new_mem = new_matches[0] if new_matches else None

        if action == "changed":

            if old_id and new_mem:

                self.sqlite.update_memory(

                    old_id,

                    status="superseded",

                    superseded_by=new_mem["id"],

                    superseded_reason="user_confirmed_change",

                )

                try:

                    self.memory.vectors.delete(old_id)

                except Exception:

                    pass

                self.memory.activate(new_mem["id"])

            self.sqlite.resolve_contradiction(event_id, "changed")

            return {"ok": True, "message": "Stored the update and preserved the earlier memory as superseded."}

        if action == "different":

            if new_mem:

                self.memory.activate(new_mem["id"])

            self.sqlite.resolve_contradiction(event_id, "different_referents")

            return {"ok": True, "message": "Kept both memories as active because they refer to different things."}

        if action == "correct_old":

            if old_id and new_mem:

                self.sqlite.update_memory(

                    old_id,

                    status="superseded",

                    superseded_by=new_mem["id"],

                    superseded_reason="user_corrected_old",

                )

                try:

                    self.memory.vectors.delete(old_id)

                except Exception:

                    pass

                self.memory.activate(new_mem["id"])

            self.sqlite.resolve_contradiction(event_id, "corrected_stored_memory")

            return {"ok": True, "message": "Corrected the stored memory. The old wording remains in history."}

        return {"ok": False, "error": "Unknown action."}
