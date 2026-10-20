# RecallBot

An AI chatbot that remembers, connects, and clarifies.

RecallBot is a **college NLP project**: a local conversational system that stores user facts, retrieves them with sentence embeddings, and flags *potential* contradictions using structured attributes, rules, and optional natural language inference (NLI). It does **not** call a paid LLM API for core chat.

## Problem statement

People say conflicting things across a conversation (“I built it in Java” later followed by “I built it in Python”). A chatbot that only appends messages cannot tell whether the user switched languages, corrected a mistake, or is talking about a different project. RecallBot treats this as an NLP problem: extract facts, retrieve related memories, estimate the relationship between two statements, and ask for clarification when confidence is low.

## Objectives

- Persist conversational facts in SQLite and a vector index.
- Retrieve paraphrased memories with semantic similarity.
- Detect likely contradictions without treating every wording change as a conflict.
- Distinguish corrections, time-based changes, compatible facts, and ambiguous references.
- Generate replies from retrieved memories and templates (no generative API required).

## Features

- Streamlit chat UI with memory retrieval badges and contradiction cards
- Memory dashboard (search, filter, correct, deactivate, delete)
- Contradiction review actions that keep historical records
- NLP insights built from live SQLite statistics
- Demo dataset that can be reset without deleting user data
- Offline evaluation set (34 labeled pairs) and pytest suite

## NLP techniques (academic notes)

**Preprocessing.** Text is Unicode-normalized and whitespace-collapsed. Negation, dates, numbers, and names are kept. Aggressive stopword removal would destroy contradiction cues such as *not* and *never*.

**Named entity recognition.** spaCy’s `en_core_web_sm` is used when installed for PERSON, ORG, GPE, DATE, and related labels. Programming languages, frameworks, projects, and preferences are **not** reliably tagged by a general NER model, so RecallBot adds lexicon and pattern extractors and stores structured `(subject, attribute, value)` records.

**Sentence embeddings and semantic similarity.** Production mode encodes text with Sentence Transformers `all-MiniLM-L6-v2`. Cosine similarity retrieves near-paraphrases (for example, “I am studying computer science” vs “What is my degree subject?”). Similarity is a **retrieval** signal. It is not proof of contradiction.

**Coreference (limited).** Pronouns such as *it*, *that project*, and *the app* are rewritten using the most recent project-like mention. This is a **lightweight context fallback**, not full coreference resolution (no mention clustering or trained coref model).

**Natural language inference.** When Hugging Face Transformers and a pretrained MNLI model load successfully, premise/hypothesis pairs are scored as contradiction, entailment, or neutral. Label IDs are mapped through `model.config.id2label` rather than assuming a fixed class order. NLI outputs are **estimates**. If the model cannot download, conservative rules still run and the UI states that advanced NLI is unavailable.

**Contradiction engine.** New facts are compared with retrieved memories using attribute/value clashes, polarity (negation), distinct project names, correction markers (`Correction:`, `I meant`), and change markers (`migrated`, `switched`). Uncertain cases ask a clarifying question instead of inventing a conflict. Old facts are **superseded**, never silently erased.

## System architecture

```
User message
    → preprocess + lightweight coreference
    → spaCy/rules NER and fact extraction
    → embed + ChromaDB/NumPy retrieval
    → contradiction detector (rules + optional NLI)
    → memory policy (store / review / supersede)
    → template response + Streamlit UI
```

| Path | Role |
| --- | --- |
| `app.py` | Streamlit entry point |
| `config.py` | Thresholds, model names, data paths |
| `database/sqlite_manager.py` | Structured memory and logs |
| `database/vector_store.py` | ChromaDB with NumPy cosine fallback |
| `nlp/` | Preprocess, NER, embeddings, NLI, retrieval, contradiction |
| `chatbot/` | Conversation and memory policies |
| `ui/` | Chat, memory, review, insights pages |
| `tests/` | Pytest suite (no large-model downloads) |
| `evaluation/` | Labeled pairs and metrics script |

## Technology stack

- Python 3.11+ (3.11 or 3.12 recommended for spaCy / PyTorch wheels)
- Streamlit, spaCy, Sentence Transformers, ChromaDB, Transformers, SQLite, Pandas, Pytest

Core chat **does not** require an OpenAI key. All user data stays under `data/` on disk.

## Installation (Windows)

In **Command Prompt** or **PowerShell**:

```bat
cd C:\Users\mahaj\OneDrive\Documents\RecallBot
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If `py -3.11` is missing, install [Python 3.11](https://www.python.org/downloads/) and retry. Python 3.14 may not have wheels for `torch`, `spacy`, or `chromadb`; the app then uses hashed embeddings, rule-based NLI, and a NumPy vector index.

Optional CPU PyTorch (if the default torch install fails):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run the Streamlit app

```bat
cd C:\Users\mahaj\OneDrive\Documents\RecallBot
.venv\Scripts\activate
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

## Run tests

Tests mock/avoid large model downloads via `RECALLBOT_TEST_MODE` and `force_fallback`.

```bat
cd C:\Users\mahaj\OneDrive\Documents\RecallBot
.venv\Scripts\activate
pytest -q
```

## Run evaluation

```bat
python evaluation\evaluate.py
```

The script uses the same detector in offline fallback mode so it does not require model downloads. **Do not quote an accuracy number unless this script has been run.** Results are written to `evaluation/last_results.json`.

## Example conversations

**Contradiction / clarification**  
User: I developed my college project using Java.  
User: I built the same project entirely using Python.  
RecallBot asks whether the language changed, this is a correction, or it is a different project.

**Compatible facts**  
User: I study computer science.  
User: I am learning Python.  
No contradiction; both memories can stay active.

**Question answering from memory**  
User: What programming language did I say I used for my project?  
RecallBot quotes the stored statement and timestamp. If nothing is similar enough: *I couldn't find a sufficiently relevant memory about that.*

**Correction**  
User: My exam is on Monday.  
User: Correction: my exam is on Tuesday.  
The Monday fact is marked superseded and kept in history.

## Model limitations

- MiniLM similarity can retrieve loosely related sentences.
- Small MNLI models mislabel challenging pairs; rules can also over- or under-fire.
- Lightweight coreference fails on nested or multi-entity stories.
- The chatbot **never invents** facts that were not stored; it also cannot “understand” like a large generative model.
- Demo data is labeled `[DEMO]` / `is_demo=1`. Use **Reset demo dataset** only after confirming the checkbox so ordinary user rows are kept.

## Troubleshooting

| Issue | What to try |
| --- | --- |
| `spacy.load` fails | `python -m spacy download en_core_web_sm` or continue without NER (rules still extract languages/tech) |
| Sentence Transformer download fails | Check network; app falls back to hashed-token embeddings |
| Transformers / CUDA errors | Keep `RECALLBOT_DEVICE=cpu` (default) |
| ChromaDB install fails on new Python | NumPy index in `data/vector_index.npz` is used automatically |
| Duplicate or missing vectors | Memory dashboard → **Rebuild / validate vector index** |
| Empty charts | Chat first or load demo data; metrics are live, not hardcoded |

## Configuration

Environment variables (optional): `RECALLBOT_EMBEDDING_MODEL`, `RECALLBOT_NLI_MODEL`, `RECALLBOT_SIM_THRESHOLD`, `RECALLBOT_TEST_MODE`. Paths are always relative to the project directory, not the current working directory of the shell.
