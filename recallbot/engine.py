"""Memory-first dialogue engine. Does not pretend to be a general-purpose LLM."""
from __future__ import annotations

import re
from .extractor import attribute_from_question, extract_facts, label_for, normalize
from .models import Reply


class RecallBot:
    def __init__(self, store, index, profile: str = "default", use_nli: bool = False):
        self.store = store
        self.index = index
        self.profile = profile
        self.use_nli = use_nli

    def _log(self, prompt: str, response: Reply) -> Reply:
        self.store.log_chat(self.profile, "user", prompt)
        self.store.log_chat(self.profile, "assistant", response.text)
        return response

    def _conflict_message(self, old: dict, new) -> str:
        message = (
            f"⚠️ I found a possible contradiction about **{label_for(new.attribute)}**.\n\n"
            f"**Earlier:** “{old['statement']}”\n\n"
            f"**Now:** “{new.statement}”\n\n"
            "Should I **update** the memory to the new information, or **keep** the earlier one? "
            "Type `update` or `keep`, or use the buttons below."
        )
        if self.use_nli:
            try:
                from .nli import classify
                nli = classify(old["statement"], new.statement)
                message += f"\n\n*NLI model: {nli['label']} ({nli['confidence']:.0%} confidence). This is supporting evidence, not the final decision.*"
            except Exception as error:
                message += f"\n\n*NLI check unavailable: {type(error).__name__}. You can still decide manually.*"
        return message

    def resolve(self, decision: str) -> Reply:
        pending = self.store.get_pending(self.profile)
        if not pending:
            return Reply("There is no contradiction waiting for your decision.")
        previous = self.store.get(self.profile, pending["existing_id"])
        fact = pending["fact"]
        if decision == "update":
            # Remove previous memory from search only after replacing in SQLite.
            if previous:
                self.store.supersede(self.profile, previous["id"])
                self.index.delete(previous["id"])
            memory = self.store.add_memory(self.profile, fact)
            self.index.upsert(memory)
            response = Reply(f"✅ Updated **{label_for(fact.attribute)}** to **{fact.value}**. The earlier fact remains in SQLite as history.", "updated")
        elif decision == "keep":
            response = Reply(f"✅ Kept the earlier memory about **{label_for(fact.attribute)}**. I did not save the new claim.", "kept")
        else:
            return Reply("Please choose `update` or `keep`.", "pending")
        self.store.clear_pending(self.profile)
        self.store.log_chat(self.profile, "assistant", response.text)
        return response

    def _remembered(self, memories: list[dict], intro: str) -> Reply:
        if not memories:
            return Reply("I don't have any saved facts about you yet. Try: **My name is Alex and I live in Pune.**")
        lines = [f"- **{label_for(m['attribute'])}:** {m['value']}" for m in memories]
        return Reply(intro + "\n\n" + "\n".join(lines), "recalled")

    def reply(self, prompt: str) -> Reply:
        prompt = prompt.strip()
        if not prompt:
            return Reply("Write a message to start chatting.")
        pending = self.store.get_pending(self.profile)
        decision = normalize(prompt)
        if pending:
            if decision in {"update", "replace", "new", "yes", "use new"}:
                self.store.log_chat(self.profile, "user", prompt)
                return self.resolve("update")
            if decision in {"keep", "old", "no", "keep old"}:
                self.store.log_chat(self.profile, "user", prompt)
                return self.resolve("keep")
            return self._log(prompt, Reply(
                "Please resolve the earlier contradiction first: type **update** to save the new fact, or **keep** to retain the old one.",
                "pending",
            ))

        q = normalize(prompt)
        if re.search(r"\b(?:what do you (?:remember|know) about me|show (?:me )?my memories|list my memories|what have i told you|summarize my profile)\b", q):
            return self._log(prompt, self._remembered(self.store.active(self.profile), "Here's what I currently remember:"))

        attr = attribute_from_question(prompt) if "?" in prompt or q.startswith(("what ", "where ", "who ", "how ", "do i ")) else None
        if attr:
            matches = self.store.by_attribute(self.profile, attr)
            if matches:
                value = matches[0]["value"]
                if attr.startswith("likes:"):
                    answer = f"According to what you told me, **{'yes' if value == 'yes' else 'no'}**: {matches[0]['statement']}"
                else:
                    answer = f"You told me your **{label_for(attr).lower()}** is **{value}**.\n\n*Source: “{matches[0]['statement']}”*"
                return self._log(prompt, Reply(answer, "recalled"))
            return self._log(prompt, Reply(f"You haven't told me your **{label_for(attr).lower()}** yet (or I haven't saved it).", "unknown"))

        facts = extract_facts(prompt)
        if facts:
            new_count = 0
            repeated = 0
            for fact in facts:
                old = self.store.by_attribute(self.profile, fact.attribute)
                if any(normalize(item["value"]) == normalize(fact.value) for item in old):
                    repeated += 1
                    continue
                if old:
                    self.store.save_pending(self.profile, old[0]["id"], fact)
                    lead = f"I saved {new_count} new fact(s).\n\n" if new_count else ""
                    return self._log(prompt, Reply(lead + self._conflict_message(old[0], fact), "pending"))
                new_memory = self.store.add_memory(self.profile, fact)
                self.index.upsert(new_memory)
                new_count += 1
            if new_count:
                return self._log(prompt, Reply(f"🧠 Saved **{new_count}** new fact(s) to long-term memory. Ask me about them later, even after restarting!", "saved"))
            if repeated:
                return self._log(prompt, Reply("I already remember that. No duplicate memory was added.", "duplicate"))

        if any(word in q for word in ("hello", "hey", "hi there", "good morning", "good evening")) or q == "hi":
            return self._log(prompt, Reply("Hello! 👋 I'm RecallBot. Tell me facts about yourself and ask me to recall them later. If you change a fact, I'll ask which version to remember."))
        if "help" in q or "what can you do" in q:
            return self._log(prompt, Reply(
                "I can **remember personal facts**, **recall them across sessions**, **search related memories**, and **flag conflicting information**. "
                "Try `My name is Alex`, `I live in Pune`, `Where do I live?`, `I live in Mumbai`, or `What do you remember about me?`"
            ))

        results = self.index.query(self.profile, prompt, limit=3)
        relevant = [r for r in results if r["score"] >= 0.38]
        if relevant:
            evidence = "\n".join(f"- “{r['statement']}” *(similarity: {r['score']:.0%})*" for r in relevant)
            return self._log(prompt, Reply("These saved memories seem related to your question:\n\n" + evidence + "\n\nI won't invent details beyond what you've told me.", "search"))
        return self._log(prompt, Reply(
            "I don't have a reliable memory to answer that. Tell me a fact like **I study at MIT-WPU**, "
            "or ask **What do you remember about me?** I'm a memory-focused NLP chatbot, not a general-purpose LLM.", "unknown"
        ))
