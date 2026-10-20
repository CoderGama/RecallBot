"""Orchestrate preprocessing, retrieval, contradiction checks, and replies."""

from __future__ import annotations

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
        retrieved = self.retriever.search(
            working_text,
            sqlite_lookup=self.sqlite.get_memory,
        )
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
            stored_ids, event_id = self._apply_policy(
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
