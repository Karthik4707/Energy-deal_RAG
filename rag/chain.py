"""Retrieval + generation with source citations."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag.ingest import get_store
from rag.prompts import CONDENSE_PROMPT, GENERAL_LABEL, GENERAL_PROMPT, SYSTEM_PROMPT
from rag.providers import get_llm

NOT_FOUND = "Not found in the documents."


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)
    general: bool = False  # True = answered from general knowledge, not the documents


def _llm(temperature: float = 0.0):
    return get_llm(temperature)


def condense(question: str, history: list[tuple[str, str]]) -> str:
    """Turn a follow-up ("and its capacity?") into a standalone question."""
    if not history:
        return question
    hist = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-4:])
    msg = CONDENSE_PROMPT.format(history=hist, question=question)
    return _llm().invoke([HumanMessage(content=msg)]).content.strip()


def page_of(doc: Document) -> str:
    """Pinecone returns numbers as floats (1.0); show them as whole pages."""
    p = doc.metadata.get("page", "?")
    try:
        return str(int(float(p)))
    except (TypeError, ValueError):
        return str(p)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', '?')}, p.{page_of(d)})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def retrieve(question: str, k: int = 4, namespace: str = "default",
             min_score: float = 0.25) -> list[Document]:
    hits = get_store(namespace).similarity_search_with_score(question, k=k)
    return [doc for doc, score in hits if score >= min_score]


def general_answer(question: str) -> Answer:
    """Fallback when the documents don't cover the question. Clearly labelled."""
    messages = [
        SystemMessage(content=GENERAL_PROMPT),
        HumanMessage(content=question),
    ]
    text = _llm(temperature=0.3).invoke(messages).content.strip()
    return Answer(f"{GENERAL_LABEL}\n\n{text}", general=True)


def ask(question: str, history: list[tuple[str, str]] | None = None,
        k: int = 4, namespace: str = "default", allow_general: bool = False) -> Answer:
    standalone = condense(question, history or [])
    docs = retrieve(standalone, k=k, namespace=namespace)
    text = NOT_FOUND
    if docs:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=format_context(docs))),
            HumanMessage(content=standalone),
        ]
        text = _llm().invoke(messages).content.strip()
    if text.startswith(NOT_FOUND):
        return general_answer(standalone) if allow_general else Answer(NOT_FOUND)
    return Answer(text, docs)
