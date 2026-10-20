def test_insert_correct_deactivate_delete(runtime):
    added = runtime.memory.add_memory(
        {
            "fact_text": "My exam is on Monday.",
            "subject": "user",
            "attribute": "exam_date",
            "value": "monday",
            "status": "active",
        }
    )
    mem_id = added["id"]
    assert runtime.sqlite.get_memory(mem_id)["status"] == "active"

    new_id = runtime.memory.correct_memory(mem_id, "My exam is on Tuesday.")
    old = runtime.sqlite.get_memory(mem_id)
    new = runtime.sqlite.get_memory(new_id)
    assert old["status"] == "superseded"
    assert new["status"] == "active"
    assert "Tuesday" in new["fact_text"]

    runtime.memory.deactivate(new_id)
    assert runtime.sqlite.get_memory(new_id)["status"] == "deleted"

    runtime.memory.delete(new_id)
    assert runtime.sqlite.get_memory(new_id) is None


def test_duplicate_prevention(runtime):
    payload = {
        "fact_text": "I study computer science.",
        "subject": "user",
        "attribute": "degree_subject",
        "value": "computer science",
        "status": "active",
    }
    first = runtime.memory.add_memory(payload)
    second = runtime.memory.add_memory(dict(payload))
    assert second["status"] == "duplicate"
    assert second["id"] == first["id"]
