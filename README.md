# RAG Document Chat

Chat with your PDF and DOCX files using LlamaIndex, ChromaDB, and OpenAI.
![img.png](img.png)

## Setup

### 1. Install dependencies

```bash
pip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-openai llama-index-llms-openai llama-index-readers-file chromadb pypdf python-docx python-dotenv rich
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add your documents

Drop any `.pdf` or `.docx` files into the `docs/` folder:

```
docs/
  my_report.pdf
  contract.docx
  research_paper.pdf
```

### 4. Run

```bash
python main.py
```

On first run the app will:
1. Load and chunk all documents in `docs/`
2. Embed the chunks using OpenAI embeddings
3. Persist the vector index to `storage/` (ChromaDB on disk)
4. Launch an interactive chat loop

On subsequent runs it loads the persisted index directly — no re-embedding needed.

---

## Commands in chat

| Input | Effect |
|-------|--------|
| Any question | Query your documents |
| `sources` | Toggle display of source chunks and scores |
| `exit` / `quit` | Exit the app |

---

## Re-indexing

When you add new documents, re-ingest everything:

```bash
# Add to existing index
python ingest.py

# Wipe and rebuild from scratch
python ingest.py --reset
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for answering |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |

---

## Project structure

```
rag_app/
├── docs/           ← Put your PDFs and DOCX files here
├── storage/        ← ChromaDB vector index (auto-created)
├── main.py         ← Chat application entry point
├── ingest.py       ← Standalone re-indexing script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Swapping the LLM

LlamaIndex supports many LLM backends. To use Claude instead of OpenAI:

```bash
pip install llama-index-llms-anthropic
```

Then in `main.py`, replace:
```python
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-4o-mini")
```
with:
```python
from llama_index.llms.anthropic import Anthropic
Settings.llm = Anthropic(model="claude-sonnet-4-20250514")
```

For a fully local setup (no API costs), use Ollama:
```bash
pip install llama-index-llms-ollama llama-index-embeddings-ollama
```

```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
Settings.llm = Ollama(model="llama3.2")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```
