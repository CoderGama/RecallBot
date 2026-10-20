"""Natural language inference with Hugging Face Transformers and a rule fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import NLI_MODEL_NAME, TEST_MODE
from nlp.lexicon import CHANGE_MARKERS, CORRECTION_MARKERS
from nlp.preprocessing import preprocess_message


@dataclass
class NLIResult:
    label: str
    confidence: float
    probs: dict[str, float]
    backend: str
    available: bool
    raw_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probs": self.probs,
            "backend": self.backend,
            "available": self.available,
            "raw_label": self.raw_label,
        }


def _normalize_label(raw: str) -> str:
    text = raw.lower().replace("_", " ").strip()
    if "contradict" in text or text in {"c", "contradiction"}:
        return "contradiction"
    if "entail" in text or text in {"e", "entailment"} or "support" in text:
        return "entailment"
    return "neutral"


class HuggingFaceNLI:
    def __init__(self, model_name: str = NLI_MODEL_NAME) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.id2label = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }
        self.backend = "transformers"
        self.torch = torch

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)[0].cpu().numpy()
        mapped: dict[str, float] = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
        for idx, value in enumerate(probs):
            raw = self.id2label.get(idx, str(idx))
            mapped[_normalize_label(raw)] += float(value)
        label = max(mapped, key=mapped.get)
        raw_idx = int(np.argmax(probs))
        return NLIResult(
            label=label,
            confidence=float(mapped[label]),
            probs=mapped,
            backend=self.backend,
            available=True,
            raw_label=self.id2label.get(raw_idx, str(raw_idx)),
        )


class RuleBasedNLI:
    """Conservative lexical/structural estimate used when Transformers is unavailable."""

    backend = "rules"

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        a = preprocess_message(premise)["lower"]
        b = preprocess_message(hypothesis)["lower"]
        if not a or not b:
            return NLIResult("neutral", 0.4, {"contradiction": 0.2, "entailment": 0.2, "neutral": 0.6}, self.backend, False)

        tokens_a = set(a.split())
        tokens_b = set(b.split())
        overlap = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))

        if any(m in b for m in CORRECTION_MARKERS):
            return NLIResult(
                "contradiction",
                0.55,
                {"contradiction": 0.55, "entailment": 0.1, "neutral": 0.35},
                self.backend,
                False,
            )
        if any(m in b for m in CHANGE_MARKERS):
            return NLIResult(
                "neutral",
                0.5,
                {"contradiction": 0.3, "entailment": 0.2, "neutral": 0.5},
                self.backend,
                False,
            )

        a_neg = " not " in f" {a} " or "n't" in a or "never" in a
        b_neg = " not " in f" {b} " or "n't" in b or "never" in b
        if overlap > 0.45 and a_neg != b_neg:
            return NLIResult(
                "contradiction",
                0.7,
                {"contradiction": 0.7, "entailment": 0.1, "neutral": 0.2},
                self.backend,
                False,
            )
        if overlap > 0.72 and a_neg == b_neg:
            return NLIResult(
                "entailment",
                min(0.8, overlap),
                {"contradiction": 0.1, "entailment": min(0.8, overlap), "neutral": 0.2},
                self.backend,
                False,
            )
        return NLIResult(
            "neutral",
            0.45 + 0.2 * overlap,
            {
                "contradiction": 0.2,
                "entailment": 0.2 + 0.3 * overlap,
                "neutral": 0.6 - 0.2 * overlap,
            },
            self.backend,
            False,
        )


class NLIClassifier:
    def __init__(self, force_fallback: bool = False) -> None:
        self.force_fallback = force_fallback or TEST_MODE
        self._impl = None
        self.warning: str | None = None

    def _get(self):
        if self._impl is not None:
            return self._impl
        if not self.force_fallback:
            try:
                self._impl = HuggingFaceNLI()
                return self._impl
            except Exception as exc:
                self.warning = (
                    "Hugging Face NLI model could not be loaded "
                    f"({exc}). Using conservative rule-based comparison."
                )
        self._impl = RuleBasedNLI()
        return self._impl

    @property
    def available(self) -> bool:
        impl = self._get()
        return getattr(impl, "backend", "") == "transformers"

    @property
    def backend(self) -> str:
        return self._get().backend

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        return self._get().predict(premise, hypothesis)
