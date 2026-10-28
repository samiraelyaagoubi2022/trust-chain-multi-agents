# scripts/measure_recall.py
"""Mesure le rappel relatif : combien d'articles du gold set sont dans le corpus final.

Usage :
    python scripts/measure_recall.py eval/gold_set_rag_eval.json data/papers.json
"""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def similar(a: str, b: str, threshold: float = 0.75) -> bool:
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def load_titles(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict) and "papers" in data:
        data = data["papers"]
    return [p.get("title", "") for p in data if p.get("title")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", help="gold set JSON {papers: [...]}")
    ap.add_argument("corpus", help="corpus final JSON (liste ou {papers: [...]})")
    ap.add_argument("--threshold", type=float, default=0.75)
    a = ap.parse_args()

    gold_titles = load_titles(a.gold)
    corpus_titles = load_titles(a.corpus)

    found, missed = [], []
    for g in gold_titles:
        match = next((c for c in corpus_titles if similar(g, c, a.threshold)), None)
        if match:
            found.append((g, match))
        else:
            missed.append(g)

    recall = len(found) / len(gold_titles) if gold_titles else 0

    print(f"Gold set : {len(gold_titles)} articles")
    print(f"Corpus final : {len(corpus_titles)} articles")
    print(f"Retrouvés : {len(found)}")
    print(f"Manqués : {len(missed)}")
    print(f"Rappel relatif : {recall:.1%}")
    print()

    if found:
        print("=== RETROUVÉS ===")
        for g, c in found:
            print(f"  ✓ {g}")
    if missed:
        print("\n=== MANQUÉS ===")
        for m in missed:
            print(f"  ✗ {m}")

    return 0


if __name__ == "__main__":
    sys.exit(main())