# ⚡ Energy Deal RAG

Ask questions about a renewable-energy M&A data room and get answers with cited sources (file + page).

> Rebuilt from scratch in Oct 2026 after the original 2024 code was lost; extended with source citations, a "not found" guardrail, retrieval evaluation and an M&A use case.

**Live demo:** _add your Streamlit Cloud link_ · **Demo GIF:** _add docs/demo.gif_

**Runs on a 100% free stack:** local FastEmbed embeddings, GPT-OSS 120B via Groq free tier, Pinecone Starter. Switch to OpenAI with two lines in `.env`.

## Why
Sell-side M&A teams answer the same buyer questions from hundreds of pages (teasers, technical reports, permits). This app finds the answer and shows exactly where it came from, so a human can verify it in seconds.

## How it works
```
PDFs ─► PyPDFLoader ─► RecursiveCharacterTextSplitter ─► FastEmbed (bge-small) ─► Pinecone (namespace per deal)
                                                                                   │
Question ─► condense with chat history ─► top-k retrieval (score threshold) ───────┘
         ─► prompt with numbered excerpts ─► GPT-OSS 120B on Groq ─► answer with [1][2] citations, or "Not found in the documents."
```

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env          # add your free Groq + Pinecone keys
python -m rag.ingest          # indexes PDFs in data/sample/
streamlit run app.py
```
Tests: `pytest` · Evaluation: `python -m eval.run_eval --k 4`

## Results
| Metric | Value |
| --- | --- |
| Retrieval hit rate @4 (15 questions) | _fill in_ |

## Sample data
`data/sample/` holds public renewable-energy reports plus one **synthetic** biogas-plant teaser (fictional company, clearly labelled) — no confidential data.

## Next steps
Hybrid search (BM25 + vectors), a reranker, per-user access control for confidential data rooms, EU-hosted models.
