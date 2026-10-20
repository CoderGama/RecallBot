def test_paraphrase_retrieval_degree(runtime):
    runtime.memory.add_memory(
        {
            "fact_text": "I am studying computer science.",
            "subject": "user",
            "attribute": "degree_subject",
            "value": "computer science",
            "status": "active",
        }
    )
    hits = runtime.retriever.search(
        "What is my degree subject?",
        sqlite_lookup=runtime.sqlite.get_memory,
        threshold=0.2,
    )
    assert hits
    assert "computer science" in hits[0]["fact_text"].lower()


def test_language_memory_retrieved_for_question(runtime):
    conv = runtime.sqlite.create_conversation()
    runtime.conversations.process(conv, "I developed my college project using Java.")
    result = runtime.conversations.process(
        conv, "What programming language did I say I used for my college project?"
    )
    assert result["retrieved"]
    assert "java" in result["assistant_text"].lower()
