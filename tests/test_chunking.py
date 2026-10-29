from langchain_core.documents import Document

from rag.chain import format_context
from rag.ingest import split


def test_split_respects_chunk_size_and_keeps_metadata():
    doc = Document(page_content="Biogas plant. " * 500, metadata={"source": "a.pdf", "page": 3})
    chunks = split([doc], chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata == {"source": "a.pdf", "page": 3} for c in chunks)


def test_format_context_numbers_sources():
    docs = [Document(page_content="2 MW", metadata={"source": "t.pdf", "page": 1})]
    assert format_context(docs).startswith("[1] (t.pdf, p.1)")
