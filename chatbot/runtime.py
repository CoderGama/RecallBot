"""Assemble SQLite, vectors, NLP components, and conversation logic."""

from __future__ import annotations

from pathlib import Path

from chatbot.conversation_manager import ConversationManager
from chatbot.memory_manager import MemoryManager
from chatbot.response_generator import ResponseGenerator
from database.sqlite_manager import SQLiteManager
from database.vector_store import VectorStore
from nlp.contradiction_detector import ContradictionDetector
from nlp.embeddings import Embedder
from nlp.nli_classifier import NLIClassifier
from nlp.semantic_retrieval import SemanticRetriever


class RecallBotRuntime:
    def __init__(
        self,
        sqlite_path: str | Path | None = None,
        chroma_path: str | Path | None = None,
        numpy_path: str | Path | None = None,
        meta_path: str | Path | None = None,
        force_fallback: bool = False,
    ) -> None:
        self.sqlite = SQLiteManager(sqlite_path)
        self.vectors = VectorStore(chroma_path, numpy_path, meta_path)
        self.embedder = Embedder(force_fallback=force_fallback)
        self.nli = NLIClassifier(force_fallback=force_fallback)
        self.memory = MemoryManager(self.sqlite, self.vectors, self.embedder)
        self.retriever = SemanticRetriever(self.embedder, self.vectors)
        self.detector = ContradictionDetector(self.nli)
        self.conversations = ConversationManager(
            self.sqlite,
            self.memory,
            self.retriever,
            self.detector,
            ResponseGenerator(),
        )

    @property
    def status(self) -> dict[str, str | None]:
        return {
            "embedding_backend": self.embedder.backend,
            "embedding_warning": self.embedder.warning,
            "nli_backend": self.nli.backend,
            "nli_warning": self.nli.warning,
            "vector_backend": self.vectors.backend,
        }
