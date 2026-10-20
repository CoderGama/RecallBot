import pytest
from recallbot.engine import RecallBot
from recallbot.extractor import attribute_from_question, extract_facts
from recallbot.retrieval import DummyIndex
from recallbot.storage import MemoryStore


@pytest.fixture
def bot():
    store = MemoryStore()
    index = DummyIndex()
    return RecallBot(store, index)


def test_extract_multiple_facts():
    facts = extract_facts("My name is Alex and I live in Pune.")
    assert [(f.attribute, f.value) for f in facts] == [("name", "Alex"), ("lives_in", "Pune")]


def test_spacy_extraction_examples():
    facts = extract_facts("I don't like cricket. I study at MIT-WPU.")
    assert (facts[0].attribute, facts[0].value) == ("likes:cricket", "no")
    assert (facts[1].attribute, facts[1].value) == ("studies_at", "MIT-WPU")


def test_question_matching():
    assert attribute_from_question("Where do I live?") == "lives_in"
    assert attribute_from_question("What is my favorite color?") == "favorite_color"
    assert attribute_from_question("Do I like cricket?") == "likes:cricket"


def test_remember_and_recall(bot):
    assert bot.reply("My name is Alex and I live in Pune.").event == "saved"
    answer = bot.reply("What is my name?")
    assert answer.event == "recalled" and "Alex" in answer.text
    assert "Pune" in bot.reply("Where do I live?").text
    assert len(bot.store.active(bot.profile)) == 2


def test_duplicate_ignored(bot):
    bot.reply("My name is Alex.")
    assert bot.reply("My name is Alex.").event == "duplicate"
    assert len(bot.store.active(bot.profile)) == 1


def test_conflict_update(bot):
    bot.reply("I live in Pune.")
    assert bot.reply("I live in Mumbai.").event == "pending"
    assert bot.store.get_pending(bot.profile)
    assert "Pune" in bot.reply("Where do I live?").text or bot.store.get_pending(bot.profile)
    answer = bot.reply("update")
    assert answer.event == "updated"
    assert "Mumbai" in bot.reply("Where do I live?").text
    records = bot.store.all(bot.profile)
    assert len(records) == 2
    assert sum(m["active"] for m in records) == 1
    assert len(bot.index.entries) == 1


def test_conflict_keep(bot):
    bot.reply("My favorite color is blue.")
    bot.reply("My favourite colour is red.")
    assert bot.reply("keep").event == "kept"
    assert "blue" in bot.reply("What is my favorite color?").text
    assert len(bot.store.all(bot.profile)) == 1


def test_preference_conflict(bot):
    bot.reply("I like cricket.")
    assert bot.reply("I don't like cricket.").event == "pending"
    bot.reply("update")
    assert "no" in bot.reply("Do I like cricket?").text


def test_profile_isolation():
    db = MemoryStore()
    index = DummyIndex()
    a = RecallBot(db, index, profile="a")
    b = RecallBot(db, index, profile="b")
    a.reply("My name is Alice.")
    b.reply("My name is Bob.")
    assert "Alice" in a.reply("What is my name?").text
    assert "Bob" in b.reply("What is my name?").text


def test_restart_sqlite_persistence(tmp_path):
    path = tmp_path / "memories.sqlite3"
    old = MemoryStore(str(path))
    index = DummyIndex()
    bot1 = RecallBot(old, index)
    bot1.reply("My name is Chris.")
    old.con.close()
    new = MemoryStore(str(path))
    index2 = DummyIndex()
    index2.sync(new.active("default"), "default")
    bot2 = RecallBot(new, index2)
    assert "Chris" in bot2.reply("What is my name?").text


def test_delete_memory(bot):
    bot.reply("My project is RecallBot.")
    target = bot.store.active(bot.profile)[0]
    bot.store.remove(bot.profile, target["id"])
    bot.index.delete(target["id"])
    assert not bot.store.active(bot.profile)
    assert bot.reply("What is my project?").event == "unknown"


def test_unknown_question_does_not_hallucinate(bot):
    assert bot.reply("What is my name?").event == "unknown"
    assert not bot.store.active(bot.profile)
