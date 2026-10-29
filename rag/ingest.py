"""Load PDFs, split into chunks, embed and upsert into Pinecone."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from rag.providers import embedding_dim, get_embeddings

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "energy-deal-rag")


def load_pdfs(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        for page in PyPDFLoader(str(p)).load():
            page.metadata["source"] = p.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1  # 1-based
            docs.append(page)
    return docs


def split(docs: list[Document], chunk_size: int = 1000, overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)


def ensure_index() -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dim = embedding_dim()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    elif pc.describe_index(INDEX_NAME).dimension != dim:
        raise SystemExit(
            f"Pinecone index '{INDEX_NAME}' has a different dimension than the embedding model ({dim}). "
            "Delete the index in the Pinecone console or set a new PINECONE_INDEX name in .env."
        )


_STORES: dict[str, PineconeVectorStore] = {}


def get_store(namespace: str = "default") -> PineconeVectorStore:
    if namespace not in _STORES:
        ensure_index()
        _STORES[namespace] = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=get_embeddings(), namespace=namespace
        )
    return _STORES[namespace]


def ingest(paths: list[Path], namespace: str = "default", chunk_size: int = 1000) -> int:
    chunks = split(load_pdfs(paths), chunk_size=chunk_size)
    get_store(namespace).add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    sample = sorted(Path("data/sample").glob("*.pdf"))
    if not sample:
        raise SystemExit("Put PDFs into data/sample/ first.")
    print(f"Ingested {ingest(sample)} chunks from {len(sample)} files.")
