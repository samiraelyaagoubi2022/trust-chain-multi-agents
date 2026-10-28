"""Génère un fichier d'annotation aveugle (labels effacés) pour la double annotation.

Usage :
    python scripts/make_blind_annotation.py eval/gold_evalset.json eval/gold_blind_15.json --sample 15 --seed 42

Le fichier de sortie contient paper_title et abstract (récupérés depuis claims_evalset.json)
pour que scripts/annotate.py puisse afficher le résumé.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="fichier gold source (avec labels)")
    ap.add_argument("dst", help="fichier de sortie (sans labels)")
    ap.add_argument("--sample", type=int, default=0, help="nombre de claims à garder (0 = tous)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.src, encoding="utf-8") as f:
        gold = json.load(f)

    # Charge les articles depuis claims_evalset.json (référencé par claims_file)
    claims_file = gold.get("claims_file", "eval/claims_evalset.json")
    with open(claims_file, encoding="utf-8") as f:
        claims_data = json.load(f)
    papers = {p["id"]: p for p in claims_data["papers"]}
    claims = {c["id"]: c for c in claims_data["claims"]}

    items = gold["annotations"]
    if a.sample and a.sample < len(items):
        ids = [x["claim_id"] for x in items]
        keep = set(random.Random(a.seed).sample(ids, a.sample))
        items = [x for x in items if x["claim_id"] in keep]

    # Enrichit chaque entrée avec paper_title et abstract
    enriched = []
    for x in items:
        cid = x["claim_id"]
        claim = claims.get(cid, {})
        paper = papers.get(claim.get("paper_id", ""), {})
        enriched.append({
            "claim_id": cid,
            "paper_id": claim.get("paper_id", ""),
            "claim": x["claim"],
            "paper_title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "human_label": "",
            "note": "",
        })

    gold["annotations"] = enriched
    with open(a.dst, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    print(f"{len(enriched)} affirmations aveugles -> {a.dst}")
    print(f"Donnez ce fichier à un second annotateur : python scripts/annotate.py {a.dst}")


if __name__ == "__main__":
    main()