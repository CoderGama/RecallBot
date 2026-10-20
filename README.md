# 🧠 RecallBot — Contradiction-Aware Conversational Memory Chatbot

A **complete local-first NLP college project**. RecallBot extracts personal facts from conversation, stores them persistently, retrieves them across restarts, and **asks before replacing contradictory facts**. No OpenAI key, paid API, or hosted LLM required.

> Important: RecallBot is a **memory-focused conversational chatbot**, not an unrestricted general-purpose AI assistant. Responses use stored evidence and rules rather than generative AI. Pattern-based extraction intentionally handles a defined set of statements; some paraphrases and complex claims will not be recognized. NLI is an *optional second opinion*, never grounds for silently overwriting a fact.

## Technologies used

| Component | Technology | What it does |
| --- | --- | --- |
| Frontend | Streamlit | Chat, profile selector, memory explorer, CSV export |
| NLP | spaCy | Tokenization, sentence splitting, optional named entity recognition |
| Semantic similarity | Sentence Transformers `all-MiniLM-L6-v2` | Encodes memories/queries as semantic vectors |
| Vector database | ChromaDB `PersistentClient` | Stores/queries semantic memories locally |
| Structured data | SQLite | Facts, history, pending conflicts and conversation history |
| NLI | Hugging Face Transformers `cross-encoder/nli-distilroberta-base` | Optional contradiction/entailment/neutral scores |
| Analytics & export | pandas | CSV export, history table |
| Testing | pytest | Automated behavioral tests |
| Language | Python 3.12 compatible | App and processing pipeline |

## Quick start (Ubuntu / Windows WSL)

Extract the ZIP and enter its folder. For example, if you copied it to your existing Windows project folder:

```bash
cd "/mnt/c/Users/mahaj/OneDrive/Documents/RecallBot"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Visit **http://localhost:8501** in the browser.

**Alternative**: inside the project folder, `bash start.sh` creates the venv, installs missing dependencies and launches Streamlit.

A Python 3.11 or 3.12 environment is recommended. A first-run internet connection is needed to download the pretrained embedding model from Hugging Face. Later runs use cached model files. **Optional** spaCy English NER model:

```bash
python -m spacy download en_core_web_sm
```

Without that extra spaCy model, the built-in English tokenizer/sentence segmenter and custom fact rules work; the NLP Inspector will not show trained NER entities.

Optional NLI: turn on the **Enable optional NLI model** switch in the left sidebar, then introduce a contradiction. It downloads the NLI model on first use and needs more RAM/disk space. The model makes prediction errors; **you** approve any update.

**WSL note:** If SQLite/ChromaDB are slow or you encounter file-lock problems while running in a OneDrive-synced Windows folder, move the project to Ubuntu's own filesystem (`~/RecallBot`) or set `RECALLBOT_DATA_DIR="$HOME/.recallbot"` before starting. Do not sync live database files with OneDrive.

## Demo script for presentation

```text
You: My name is Alex and I live in Pune.
Bot: Saved 2 new facts to long-term memory.
You: What is my name?
Bot: You told me your name is Alex.
You: Where do I live?
Bot: You told me your current city/location is Pune.
You: I live in Mumbai.
Bot: Possible contradiction: earlier Pune, now Mumbai. Update or keep?
You: update
Bot: Updated current city/location to Mumbai.
You: Where do I live?
Bot: You told me your current city/location is Mumbai.
You: What do you remember about me?
Bot: Name: Alex; Current city/location: Mumbai.
```

Reload the webpage or restart the server and ask again; facts persist in SQLite and semantic vectors in ChromaDB.

Additional examples:

- `My favorite color is blue.` → `What is my favorite color?`
- `I study at MIT-WPU.` → `Where do I study?`
- `I like cricket.` → `Do I like cricket?`
- `I don't like cricket.` → conflicting preference, ask update/keep
- `My project is RecallBot.` → `What is my project?`
- `Tell me about my studies.` → similarity-based related memories if sufficiently similar

## How it works

```text
Streamlit user input
       |
       v
spaCy tokenizer + sentence splitter + explicit fact-pattern extractor
       |
       +-- Recognized fact? -- SQLite: compare same normalized attribute
       |                        |-- duplicate -> ignore
       |                        |-- new -> save SQLite + embed in ChromaDB
       |                        '-- different -> optional NLI + user confirmation
       |                                             |-- update -> supersede old
       |                                             '-- keep -> discard new
       |
       '-- Question? -> direct attribute lookup in SQLite
                            or Sentence Transformer vector similarity in ChromaDB
                         -> grounded reply with source evidence
```

Contradictions are scoped by **attribute** (`lives_in`, `favorite_color`, `likes:cricket`, ...), not naive keyword matching. Semantic search is used for recall, but **a high cosine similarity alone does not prove a contradiction**. No change to saved memories happens without a confirmation. The new fact is held in the `pending` table until you decide.

### Folder structure

```text
RecallBot/
├── app.py                     # Streamlit UI
├── requirements.txt
├── start.sh                   # One-command launch on Ubuntu/WSL
├── recallbot/
│   ├── models.py              # Fact and reply dataclasses
│   ├── extractor.py           # spaCy + pattern recognition
│   ├── storage.py             # SQLite persistence
│   ├── retrieval.py           # MiniLM + ChromaDB
│   ├── nli.py                 # Optional Hugging Face NLI
│   └── engine.py              # Dialogue and contradiction flow
├── tests/
│   └── test_recallbot.py
└── data/                      # Databases generated at runtime
```

## Run tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Tests use a deterministic in-memory semantic-index stub to run without downloading pretrained models. The production application uses the real ChromaDB and Sentence Transformers implementations.

## Accuracy, privacy and limitations

- All profile facts, chat and vectors are stored under `data/` (or `RECALLBOT_DATA_DIR`); the app does not encrypt these files or implement authenticated multiuser accounts. Use on a trusted local computer; do not expose the Streamlit server to the public internet.
- The first start downloads models; enabling NLI downloads another model. These are not paid APIs.
- Facts are collected with explicit patterns and can be incomplete. More complex inference, multilingual conversation and open-ended generative Q&A are **not** implemented.
- The spaCy tokenizer is always used for parsing; the trained spaCy NER model is optional and only used in NLP Inspector.
- Contradictions are recognized when two explicit facts have the same attribute and different values. These may be updates rather than logical contradictions; the user resolves them.
- Numerical and name value changes, nicknames, synonyms, time-dependent changes, and multi-valued attributes may need more sophisticated ontology/coreference logic.
- Memory profiles are convenience namespaces, **not security isolation**. Do not store secrets.
- If a model or vector DB is unavailable, the UI provides an error; tests alone do not prove that model downloads will succeed in your WSL network.

## Troubleshooting

- `python: command not found` before venv activation → use `python3 -m venv .venv`.
- `ModuleNotFoundError` → activate venv, then `python -m pip install -r requirements.txt`.
- `Address already in use` → use `python -m streamlit run app.py --server.port 8502`.
- `Model download` or `SSL` errors → verify that WSL internet access works and retry; models are hosted on Hugging Face.
- `sqlite3.OperationalError: database is locked` → move data to Linux filesystem: `export RECALLBOT_DATA_DIR="$HOME/.recallbot"`; close other app instances.
