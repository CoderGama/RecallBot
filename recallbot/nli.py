"""Optional pretrained Natural Language Inference check, downloaded on demand."""
from functools import lru_cache

MODEL_NAME = "cross-encoder/nli-distilroberta-base"
LABELS = ("contradiction", "entailment", "neutral")


@lru_cache(maxsize=1)
def load_nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def classify(old: str, new: str) -> dict:
    import torch
    tokenizer, model = load_nli()
    tokens = tokenizer(old, new, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**tokens).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()
    index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return {"label": LABELS[index], "confidence": round(probabilities[index], 3)}
