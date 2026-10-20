def _rel(runtime, first: str, second: str) -> str:
    conv = runtime.sqlite.create_conversation()
    runtime.conversations.process(conv, first)
    out = runtime.conversations.process(conv, second)
    return out["detection"]["relation"]


def test_favorite_language_contradiction(runtime):
    relation = _rel(
        runtime,
        "My favorite programming language is Java.",
        "My favorite programming language is Python.",
    )
    assert relation == "contradiction"


def test_compatible_study_and_learning(runtime):
    relation = _rel(
        runtime,
        "I study computer science.",
        "I am learning Python.",
    )
    assert relation in {"neutral", "entailment"}


def test_different_projects_not_contradiction(runtime):
    relation = _rel(
        runtime,
        "Project Alpha uses Java.",
        "Project Beta uses Python.",
    )
    assert relation in {"neutral", "entailment"}


def test_explicit_correction(runtime):
    conv = runtime.sqlite.create_conversation()
    runtime.conversations.process(conv, "My exam is on Monday.")
    out = runtime.conversations.process(conv, "Correction: my exam is on Tuesday.")
    assert out["detection"]["relation"] == "correction"
    memories = runtime.sqlite.list_memories()
    statuses = {m["status"] for m in memories}
    assert "superseded" in statuses
    assert any("Tuesday" in m["fact_text"] and m["status"] == "active" for m in memories)


def test_negation_contradiction(runtime):
    relation = _rel(
        runtime,
        "I have completed the project.",
        "I have not completed the project.",
    )
    assert relation == "contradiction"


def test_change_over_time(runtime):
    relation = _rel(
        runtime,
        "I use Java for this project.",
        "I migrated the project to Python.",
    )
    assert relation == "change_over_time"


def test_ambiguous_reference_asks(runtime):
    conv = runtime.sqlite.create_conversation()
    runtime.conversations.process(conv, "I built a chatbot using Java.")
    out = runtime.conversations.process(conv, "It now uses Python.")
    assert out["detection"]["relation"] in {"change_over_time", "ambiguous", "contradiction"}
    assert out["detection"]["should_ask"] is True


def test_no_memory_question(runtime):
    conv = runtime.sqlite.create_conversation()
    out = runtime.conversations.process(conv, "What is my favorite food?")
    assert "couldn't find" in out["assistant_text"].lower()
