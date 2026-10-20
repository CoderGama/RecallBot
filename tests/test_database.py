from chatbot.runtime import RecallBotRuntime


def test_sqlite_persists_across_connections(tmp_path):
    path = tmp_path / "persist.sqlite3"
    first = RecallBotRuntime(
        sqlite_path=path,
        chroma_path=tmp_path / "c1",
        numpy_path=tmp_path / "n1.npz",
        meta_path=tmp_path / "n1.json",
        force_fallback=True,
    )
    conv = first.sqlite.create_conversation("persist")
    mem_id = first.sqlite.insert_memory(
        {
            "fact_text": "I study computer science.",
            "subject": "user",
            "attribute": "degree_subject",
            "value": "computer science",
            "status": "active",
        }
    )
    first.sqlite.add_message(conv, "user", "hello")

    second = RecallBotRuntime(
        sqlite_path=path,
        chroma_path=tmp_path / "c1",
        numpy_path=tmp_path / "n1.npz",
        meta_path=tmp_path / "n1.json",
        force_fallback=True,
    )
    loaded = second.sqlite.get_memory(mem_id)
    assert loaded is not None
    assert loaded["value"] == "computer science"
    assert len(second.sqlite.list_messages(conv)) == 1


def test_vector_sqlite_sync_rebuild(runtime):
    added = runtime.memory.add_memory(
        {
            "fact_text": "My favorite programming language is Java.",
            "subject": "user",
            "attribute": "favorite_language",
            "value": "java",
            "status": "active",
        }
    )
    mem_id = added["id"]
    assert runtime.vectors.contains(mem_id) or runtime.vectors.count() >= 1
    runtime.memory.delete(mem_id)
    assert runtime.sqlite.get_memory(mem_id) is None
    runtime.memory.add_memory(
        {
            "fact_text": "I live in Pune.",
            "subject": "user",
            "attribute": "location",
            "value": "pune",
            "status": "active",
        }
    )
    report = runtime.memory.rebuild_index()
    assert report["rebuilt"] >= 1
    assert report["sqlite_active"] == report["rebuilt"] or report["consistent"]
