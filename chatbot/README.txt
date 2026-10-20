RecallBot project fix - conversation_manager.py

1. Back up your existing RecallBot/chatbot/conversation_manager.py.
2. Copy the included conversation_manager.py into RecallBot/chatbot/ and replace the old file.
3. From project root run python -m pytest -q.

The fix normalizes explicit named-project facts and excludes semantic search matches from unrelated projects before contradiction detection.
