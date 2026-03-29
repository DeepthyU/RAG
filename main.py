"""
RAG Application — Chat with your PDF/DOCX documents
Uses LlamaIndex + ChromaDB (local vector store) + Ollama (fully local LLM)
"""

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

DOCS_DIR = Path("docs")
STORAGE_DIR = Path("storage")
COLLECTION_NAME = "rag_documents"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K", 5))

console = Console()


# ── LlamaIndex global settings ────────────────────────────────────────────────
def configure_llm():
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


# ── Vector store ──────────────────────────────────────────────────────────────
def get_vector_store() -> ChromaVectorStore:
    STORAGE_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


# ── Index: build or load ──────────────────────────────────────────────────────
def build_index(vector_store: ChromaVectorStore) -> VectorStoreIndex:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        console.print(
            f"[bold red]No documents found.[/] "
            f"Add PDF or DOCX files to [cyan]{DOCS_DIR}/[/] and re-run."
        )
        sys.exit(1)

    console.print(f"[dim]Loading documents from [cyan]{DOCS_DIR}/[/]…[/]")

    documents = SimpleDirectoryReader(
        str(DOCS_DIR),
        required_exts=[".pdf", ".docx"],
        recursive=True,
    ).load_data()

    console.print(f"[dim]Loaded [bold]{len(documents)}[/] document pages/sections.[/]")
    console.print(
        f"[dim]Embedding with [cyan]{EMBED_MODEL}[/] via Ollama… "
        f"(this may take a few minutes on first run)[/]"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    console.print("[green]✓ Index built and persisted to storage/[/]")
    return index


def load_index(vector_store: ChromaVectorStore) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )


# ── Query engine ──────────────────────────────────────────────────────────────
def get_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
    )


# ── Ollama health check ───────────────────────────────────────────────────────
def check_ollama():
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    except urllib.error.URLError:
        console.print(
            f"[bold red]Cannot reach Ollama at {OLLAMA_BASE_URL}[/]\n"
            "Make sure Ollama is installed and running:\n"
            "  [cyan]ollama serve[/]\n\n"
            "Then pull the required models:\n"
            f"  [cyan]ollama pull {LLM_MODEL}[/]\n"
            f"  [cyan]ollama pull {EMBED_MODEL}[/]"
        )
        sys.exit(1)


# ── CLI chat loop ─────────────────────────────────────────────────────────────
def chat_loop(query_engine):
    console.print(
        Panel(
            f"[bold]RAG Document Chat[/] [dim]· {LLM_MODEL} via Ollama[/]\n"
            "[dim]Ask anything about your documents. "
            "Type [cyan]exit[/] or [cyan]quit[/] to stop, "
            "[cyan]sources[/] to toggle source display.[/]",
            border_style="cyan",
        )
    )

    show_sources = False

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/]")
            break

        if question.lower() == "sources":
            show_sources = not show_sources
            state = "on" if show_sources else "off"
            console.print(f"[dim]Source display turned {state}.[/]")
            continue

        with console.status(f"[dim]Thinking with {LLM_MODEL}…[/]"):
            response = query_engine.query(question)

        console.print("\n[bold green]Assistant[/]")
        console.print(Markdown(str(response)))

        if show_sources and response.source_nodes:
            console.print("\n[dim]─── Sources ───[/]")
            for i, node in enumerate(response.source_nodes, 1):
                meta = node.metadata
                filename = meta.get("file_name", "unknown")
                page = meta.get("page_label", meta.get("page", "?"))
                score = f"{node.score:.3f}" if node.score is not None else "n/a"
                console.print(
                    f"  [dim]{i}. {filename} · page {page} · score {score}[/]"
                )


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    check_ollama()
    configure_llm()

    vector_store = get_vector_store()
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    has_existing = collection.count() > 0

    if has_existing:
        console.print(
            f"[dim]Found existing index with "
            f"[bold]{collection.count()}[/] vectors. Loading…[/]"
        )
        index = load_index(vector_store)
    else:
        index = build_index(vector_store)

    query_engine = get_query_engine(index)
    chat_loop(query_engine)


if __name__ == "__main__":
    main()