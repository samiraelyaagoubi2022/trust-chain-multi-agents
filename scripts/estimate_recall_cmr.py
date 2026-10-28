# scripts/estimate_recall_cmr.py
"""Estimation du rappel par Capture-Mark-Recapture (Lincoln-Petersen).

Usage :
    python scripts/estimate_recall_cmr.py papers_arxiv.json papers_openalex.json --union papers.json
"""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def similar(a: str, b: str, threshold: float = 0.85) -> bool:
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
    ap.add_argument("search_a", help="résultats de la recherche 1 (JSON)")
    ap.add_argument("search_b", help="résultats de la recherche 2 (JSON)")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--union", help="corpus final (optionnel) pour le rappel réel")
    a = ap.parse_args()

    titles_a = load_titles(a.search_a)
    titles_b = load_titles(a.search_b)
    n1, n2 = len(titles_a), len(titles_b)

    common = 0
    for ta in titles_a:
        if any(similar(ta, tb, a.threshold) for tb in titles_b):
            common += 1
    m = common

    print(f"Recherche 1 : {n1} articles")
    print(f"Recherche 2 : {n2} articles")
    print(f"Chevauchement : {m} articles")
    print()

    if m == 0:
        print("[WARNING]  Aucun chevauchement : impossible d'estimer N.")
        return 1

    N_est = ((n1 + 1) * (n2 + 1) / (m + 1)) - 1
    union = n1 + n2 - m
    recall_est = union / N_est if N_est > 0 else 0

    print("=== ESTIMATION CMR (Lincoln-Petersen, correction Chapman) ===")
    print(f"Total estimé d'articles existants : N ≈ {N_est:.0f}")
    print(f"Union des deux recherches : {union}")
    print(f"Rappel estimé : {recall_est:.1%}")
    print()

    if a.union:
        union_titles = load_titles(a.union)
        found = 0
        for tu in union_titles:
            if any(similar(tu, ta, a.threshold) for ta in titles_a) or \
               any(similar(tu, tb, a.threshold) for tb in titles_b):
                found += 1
        print(f"=== RAPPEL RÉEL (sur le corpus final) ===")
        print(f"Corpus final : {len(union_titles)} articles")
        print(f"Dont présents dans A ou B : {found}")

    return 0


if __name__ == "__main__":
    sys.exit(main())