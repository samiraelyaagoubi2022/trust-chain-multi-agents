# scripts/measure_variance.py
"""Mesure la variance entre plusieurs runs du pipeline.

Calcule la Jaccard similarity entre les corpus de plusieurs runs
(pour vérifier la stabilité de la recherche).

Usage :
    python scripts/measure_variance.py eval/results/final/papers_run1.json eval/results/final/papers_run2.json eval/results/final/papers_run3.json
"""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations


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


def jaccard(titles_a: list[str], titles_b: list[str], threshold: float) -> float:
    """Jaccard similarity entre deux listes de titres."""
    set_a = set()
    for t in titles_a:
        set_a.add(norm(t))
    set_b = set()
    for t in titles_b:
        set_b.add(norm(t))
    # match approximatif
    inter = 0
    for ta in titles_a:
        if any(similar(ta, tb, threshold) for tb in titles_b):
            inter += 1
    union = len(titles_a) + len(titles_b) - inter
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="fichiers de runs (>= 2)")
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()

    if len(a.runs) < 2:
        sys.exit("Il faut au moins 2 runs pour comparer.")

    corpora = {r: load_titles(r) for r in a.runs}
    for r, titles in corpora.items():
        print(f"{r} : {len(titles)} articles")
    print()

    print("=== JACCARD SIMILARITY (paires) ===")
    scores = []
    for r1, r2 in combinations(a.runs, 2):
        j = jaccard(corpora[r1], corpora[r2], a.threshold)
        scores.append(j)
        print(f"  J({r1.split('/')[-1]}, {r2.split('/')[-1]}) = {j:.2f}")

    print()
    print(f"Moyenne : {sum(scores) / len(scores):.2f}")
    print(f"Min : {min(scores):.2f} | Max : {max(scores):.2f}")
    print()
    if sum(scores) / len(scores) >= 0.7:
        print("[OK] Variance faible : le pipeline est stable entre les runs.")
    elif sum(scores) / len(scores) >= 0.5:
        print("[WARNING]  Variance modérée : le pipeline est moyennement stable.")
    else:
        print("[ERROR] Variance élevée : le pipeline n'est pas stable. Vérifiez les requêtes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())