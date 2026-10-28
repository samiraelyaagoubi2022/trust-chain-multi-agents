# scripts/make_blind_sample.py
"""Crée un échantillon aveugle pour la double annotation.

Croise deux fichiers :
  - gold_merged.json    : labels humains (claim_id -> human_label)
  - claims_merged.json  : claims + titres + abstracts (claim_id -> texte complet)

Usage :
    python scripts/make_blind_sample.py eval/gold_merged.json eval/claims_merged.json \
        eval/blind_sample.json --n 15 --seed 42
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", help="annotations humaines (claim_id, human_label, ...)")
    ap.add_argument("claims", help="claims complets (claim, paper_title, abstract, ...)")
    ap.add_argument("out", help="fichier aveugle à produire")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # 1. labels humains : claim_id -> label
    with open(a.gold, encoding="utf-8") as f:
        gold = json.load(f)
    gold_ann = gold["annotations"] if isinstance(gold, dict) else gold
    labels = {x["claim_id"]: x.get("human_label", "") for x in gold_ann}

    # 2. claims complets : claim_id -> dict complet
    with open(a.claims, encoding="utf-8") as f:
        claims_data = json.load(f)
    papers_by_id = {p["id"]: p for p in claims_data.get("papers", [])}
    claims_by_id = {}
    for c in claims_data.get("claims", []):
        pid = c.get("paper_id", "")
        paper = papers_by_id.get(pid, {})
        claims_by_id[c["id"]] = {
            "claim_id": c["id"],
            "claim": c.get("text") or c.get("claim", ""),
            "paper_id": pid,
            "paper_title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
        }

    # 3. intersection : on ne garde que les claims qui ont un label ET un texte
    common = [cid for cid in claims_by_id if cid in labels]
    print(f"{len(common)} claims communs entre gold et claims_merged")

    if not common:
        raise SystemExit("Aucun claim_id commun : vérifiez que les deux fichiers "
                         "utilisent les mêmes identifiants.")

    sample_ids = random.Random(a.seed).sample(common, min(a.n, len(common)))
    blind = []
    for cid in sample_ids:
        item = dict(claims_by_id[cid])
        item["human_label"] = ""   # aveugle
        blind.append(item)

    # contrôle de sanité
    missing = [b["claim_id"] for b in blind
               if not b["paper_title"] or not b["abstract"] or not b["claim"]]
    if missing:
        print(f"ATTENTION : {len(missing)} claims sans titre/abstract/claim : {missing[:5]}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"source": a.gold, "annotations": blind}, f, ensure_ascii=False, indent=2)

    print(f"{len(blind)} claims à annoter en aveugle -> {a.out}")
    print(f"Commande à donner à votre camarade :")
    print(f"    python scripts/annotate.py {a.out}")


if __name__ == "__main__":
    main()