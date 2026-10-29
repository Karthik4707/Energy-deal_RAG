"""Streamlit chat UI: ask questions over a renewable-energy deal data room."""
import tempfile
from pathlib import Path

import streamlit as st

from rag.chain import ask
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
    if st.button("Clear chat"):
        st.session_state.history = []

st.session_state.setdefault("history", [])

for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

if question := st.chat_input("e.g. What is the installed capacity of the plant?"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Searching the documents..."):
        answer = ask(question, st.session_state.history, k=top_k, namespace=namespace)
        st.write(answer.text)
        if answer.sources:
            st.markdown("**Sources**")
            for i, d in enumerate(answer.sources, start=1):
                st.markdown(f"[{i}] `{d.metadata.get('source')}` – page {d.metadata.get('page')}")
                if show_chunks:
                    st.caption(d.page_content[:600] + "…")
    st.session_state.history.append((question, answer.text))
