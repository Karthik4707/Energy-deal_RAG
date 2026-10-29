"""Streamlit chat UI: ask questions over a renewable-energy deal data room."""
import tempfile
from pathlib import Path

import streamlit as st

from rag.chain import ask, page_of
from rag.ingest import ingest

st.set_page_config(page_title="Energy Deal RAG", page_icon="⚡", layout="wide")
st.title("⚡ Energy Deal RAG")
st.caption("Ask questions about a renewable-energy deal data room. Every answer cites its sources.")

with st.sidebar:
    st.header("Data room")
    namespace = st.text_input("Namespace (one per deal)", value="default")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, 100)
    if st.button("Index uploaded files", disabled=not files):
        with st.spinner("Chunking and embedding..."), tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in files:
                p = Path(tmp) / f.name
                p.write_bytes(f.getvalue())
                paths.append(p)
            n = ingest(paths, namespace=namespace, chunk_size=chunk_size)
        st.success(f"Indexed {n} chunks from {len(paths)} file(s).")
    st.divider()
    top_k = st.slider("Top-k chunks", 1, 10, 4)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    allow_general = st.toggle(
        "Allow general-knowledge answers",
        value=False,
        help="Off = strict: answers only from the documents. "
             "On = if the documents don't cover it, answer from general knowledge, clearly labelled.",
    )
    if st.button("Clear chat"):
        st.session_state.turns = []

st.session_state.setdefault("turns", [])  # list of (question, Answer)


def show_answer(answer) -> None:
    st.markdown(answer.text)
    if answer.sources:
        st.markdown("**Sources**")
        for i, d in enumerate(answer.sources, start=1):
            st.markdown(f"[{i}] `{d.metadata.get('source')}` – page {page_of(d)}")
            if show_chunks:
                st.caption(d.page_content[:600] + "…")


for q, a in st.session_state.turns:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        show_answer(a)

if question := st.chat_input("e.g. What is the installed capacity of the plant?"):
    st.chat_message("user").write(question)
    history = [(q, a.text) for q, a in st.session_state.turns]
    with st.chat_message("assistant"), st.spinner("Searching the documents..."):
        answer = ask(question, history, k=top_k, namespace=namespace, allow_general=allow_general)
        show_answer(answer)
    st.session_state.turns.append((question, answer))
