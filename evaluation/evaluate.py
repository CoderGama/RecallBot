"""Offline evaluation of contradiction relations and retrieval (no paid APIs)."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RECALLBOT_TEST_MODE", "1")

from chatbot.runtime import RecallBotRuntime  # noqa: E402
from nlp.entity_extraction import extract_facts  # noqa: E402

LABELS = [
    "contradiction",
    "entailment",
    "neutral",
    "correction",
    "change_over_time",
    "ambiguous",
]


def load_dataset() -> list[dict]:
    path = ROOT / "evaluation" / "sample_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def predict_pair(runtime: RecallBotRuntime, a: str, b: str) -> dict:
    conv = runtime.sqlite.create_conversation()
    runtime.conversations.process(conv, a)
    result = runtime.conversations.process(conv, b)
    return result


def precision_recall_f1(y_true: list[str], y_pred: list[str], label: str) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    table = {gold: {lab: 0 for lab in LABELS} for gold in LABELS}
    for t, p in zip(y_true, y_pred):
        gold = t if t in table else "neutral"
        pred = p if p in table[gold] else "neutral"
        table[gold][pred] += 1
    return table


def main() -> None:
    tmp = ROOT / "data" / "eval_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    y_true: list[str] = []
    y_pred: list[str] = []
    retrieval_tp = 0
    retrieval_fp = 0
    retrieval_fn = 0
    details = []

    for i, item in enumerate(dataset):
        runtime = RecallBotRuntime(
            sqlite_path=tmp / f"e{i}.sqlite3",
            chroma_path=tmp / f"e{i}_chroma",
            numpy_path=tmp / f"e{i}.npz",
            meta_path=tmp / f"e{i}.json",
            force_fallback=True,
        )
        out = predict_pair(runtime, item["statement_a"], item["statement_b"])
        pred = out["detection"]["relation"]
        gold = item["expected_label"]
        y_true.append(gold)
        y_pred.append(pred)
        details.append(
            {
                "id": item["id"],
                "gold": gold,
                "pred": pred,
                "correct": gold == pred or (gold == "ambiguous" and pred in {"ambiguous", "change_over_time", "contradiction"}),
            }
        )

        query = item.get("paraphrase_query")
        if query:
            hits = runtime.retriever.search(query, sqlite_lookup=runtime.sqlite.get_memory, threshold=0.15)
            if hits:
                retrieval_tp += 1
            else:
                retrieval_fn += 1
            # A distractor query should not always retrieve
        facts_b = extract_facts(item["statement_b"])
        if item["category"] == "neutral" and out["detection"]["relation"] == "contradiction":
            retrieval_fp += 1
        _ = facts_b

    # Retrieval scores over paraphrase items only
    para_items = [x for x in dataset if x.get("paraphrase_query")]
    retrieval_precision = retrieval_tp / max(1, retrieval_tp + retrieval_fp)
    retrieval_recall = retrieval_tp / max(1, retrieval_tp + retrieval_fn) if para_items else 0.0

    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    relaxed = sum(1 for row in details if row["correct"])
    acc = exact / len(y_true)
    relaxed_acc = relaxed / len(y_true)

    print("RecallBot evaluation")
    print(f"Examples: {len(dataset)}")
    print(f"NLI backend: rule-based fallback (force_fallback=True for reproducible offline eval)")
    print(f"Accuracy (exact label match): {acc:.3f}")
    print(
        "Relaxed accuracy (ambiguous may match change_over_time/contradiction): "
        f"{relaxed_acc:.3f}"
    )
    print()
    print("Per-class precision / recall / F1")
    f1s = []
    for label in LABELS:
        p, r, f1 = precision_recall_f1(y_true, y_pred, label)
        f1s.append(f1)
        support = Counter(y_true)[label]
        print(f"  {label:20s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  n={support}")
    print(f"Macro F1: {sum(f1s)/len(f1s):.3f}")
    print()
    print("Confusion matrix (rows = gold, columns = predicted)")
    table = confusion(y_true, y_pred)
    header = "gold\\\\pred".ljust(20) + "".join(lab[:8].ljust(10) for lab in LABELS)
    print(header)
    for gold in LABELS:
        row = gold[:18].ljust(20) + "".join(str(table[gold][lab]).ljust(10) for lab in LABELS)
        print(row)
    print()
    print(
        f"Semantic retrieval (paraphrase queries={len(para_items)}): "
        f"P={retrieval_precision:.3f} R={retrieval_recall:.3f}"
    )
    print()
    print("Mismatches")
    for row in details:
        if row["gold"] != row["pred"]:
            print(f"  {row['id']}: gold={row['gold']} pred={row['pred']}")

    out_path = ROOT / "evaluation" / "last_results.json"
    out_path.write_text(
        json.dumps(
            {
                "accuracy": acc,
                "relaxed_accuracy": relaxed_acc,
                "macro_f1": sum(f1s) / len(f1s),
                "retrieval_precision": retrieval_precision,
                "retrieval_recall": retrieval_recall,
                "details": details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
