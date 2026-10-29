"""Model providers. Default is a 100% free stack:
- Embeddings: FastEmbed (BAAI/bge-small-en-v1.5, 384 dims) runs locally on CPU, no API key.
- Chat LLM: Groq free tier (no credit card), e.g. openai/gpt-oss-120b.
Set EMBED_PROVIDER=openai / LLM_PROVIDER=openai in .env to switch to OpenAI instead.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "fastembed").lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_EMBED_DEFAULTS = {
    "fastembed": ("BAAI/bge-small-en-v1.5", 384),
    "openai": ("text-embedding-3-small", 1536),
}


def embedding_dim() -> int:
    return _EMBED_DEFAULTS[EMBED_PROVIDER][1]


def get_embeddings():
    model = os.getenv("EMBED_MODEL") or _EMBED_DEFAULTS[EMBED_PROVIDER][0]
    if EMBED_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=model)


def get_llm(temperature: float = 0.0):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"), temperature=temperature)
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.getenv("CHAT_MODEL", "openai/gpt-oss-120b"), temperature=temperature)
