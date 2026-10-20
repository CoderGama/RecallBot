"""Load labeled demo conversations without touching real user memories."""

from __future__ import annotations

from chatbot.runtime import RecallBotRuntime
from config import DEMO_TAG

DEMO_TURNS = [
    ("I developed my college project using Java.", False),
    ("I built the same project entirely using Python.", False),
    ("My favorite programming language is Java.", False),
    ("My favorite programming language is Python.", False),
    ("I study computer science.", False),
    ("I am learning Python.", False),
    ("Project Alpha uses Java.", False),
    ("Project Beta uses Python.", False),
    ("My exam is on Monday.", False),
    ("Correction: my exam is on Tuesday.", False),
    ("I have completed the project.", False),
    ("I have not completed the project.", False),
    ("I use Java for this project.", False),
    ("I migrated the project to Python.", False),
    ("I built a chatbot using Java.", False),
    ("It now uses Python.", False),
    ("What programming language did I say I used for my college project?", True),
]


def load_demo(runtime: RecallBotRuntime) -> str:
    conv_id = runtime.sqlite.create_conversation(title=f"{DEMO_TAG} sample session", is_demo=True)
    for text, _is_q in DEMO_TURNS:
        runtime.conversations.process(conv_id, text, is_demo=True)
    return conv_id


def reset_demo(runtime: RecallBotRuntime) -> dict:
    deleted = runtime.sqlite.delete_demo_data()
    for mem_id in deleted:
        try:
            runtime.vectors.delete(mem_id)
        except Exception:
            pass
    report = runtime.memory.rebuild_index()
    report["deleted_demo_memories"] = len(deleted)
    return report
