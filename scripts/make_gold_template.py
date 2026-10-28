"""Crée le fichier d'annotation humaine (gold) à partir d'un fichier de claims.

    python scripts/make_gold_template.py eval/claims_strong.json eval/gold_strong.json

Remplissez `human_label` avec : etayee | partielle | non_etayee
Annotez AVANT de lancer le vérificateur, pour ne pas être influencé.
"""
import json
import sys


def main(src: str, dst: str):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    papers = {p["id"]: p for p in data["papers"]}
    items = []
    for c in data["claims"]:
        p = papers[c["paper_id"]]
        items.append({"claim_id": c["id"], "paper_id": c["paper_id"],
                      "claim": c["text"], "paper_title": p["title"],
                      "abstract": p["abstract"],
                      "human_label": "",          # etayee | partielle | non_etayee
                      "note": ""})
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"claims_file": src, "annotations": items}, f, ensure_ascii=False, indent=2)
    print(f"{len(items)} affirmations -> {dst}. Remplissez 'human_label'.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])