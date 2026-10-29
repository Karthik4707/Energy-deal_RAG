"""Retrieval hit rate: does the expected source/page appear in the top-k chunks?

eval/questions.csv columns: question,expected_source,expected_page,expected_answer
Run: python -m eval.run_eval --k 4
"""
import argparse
import csv
from pathlib import Path

from rag.chain import page_of, retrieve


def main(k: int) -> None:
    rows = list(csv.DictReader(Path("eval/questions.csv").open(encoding="utf-8")))
    hits = 0
    for r in rows:
        docs = retrieve(r["question"], k=k, min_score=0.0)
        found = any(
            d.metadata.get("source") == r["expected_source"]
            and page_of(d) == r["expected_page"].strip()
            for d in docs
        )
        hits += found
        print(f"{'HIT ' if found else 'MISS'} | {r['question']}")
    print(f"\nHit rate @{k}: {hits}/{len(rows)} = {hits / max(len(rows), 1):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    main(ap.parse_args().k)
