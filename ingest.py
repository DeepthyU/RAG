"""
ingest.py — Re-index documents without launching the chat loop.
Useful when you add new files and want to refresh the index.

Usage:
    python ingest.py            # index everything in docs/
    python ingest.py --reset    # wipe the index first, then re-index
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rich.console import Console

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DOCS_DIR = Path("docs")
STORAGE_DIR = Path("storage")
COLLECTION_NAME = "rag_documents"

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing index before re-indexing.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/] OPENAI_API_KEY not set.")
        sys.exit(1)

    if args.reset and STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        console.print("[yellow]Existing index wiped.[/]")

    Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    Settings.embed_model = OpenAIEmbedding(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    Settings.node_parser = SentenceSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
    )

    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        console.print(f"[bold red]No documents found in {DOCS_DIR}/[/]")
        sys.exit(1)

    documents = SimpleDirectoryReader(
        str(DOCS_DIR),
        required_exts=[".pdf", ".docx"],
        recursive=True,
    ).load_data()

    console.print(f"Loaded [bold]{len(documents)}[/] pages/sections.")

    STORAGE_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    console.print(f"[green]✓ Done. {collection.count()} vectors stored in storage/[/]")


if __name__ == "__main__":
    main()
