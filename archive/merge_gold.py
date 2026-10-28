"""Fusionne plusieurs fichiers gold en un seul (déduplication par claim_id).

Usage :
    python scripts/merge_gold.py eval/gold_evalset.json eval/gold_more.json --out eval/gold_merged.json

Règles de fusion :
- Si un claim_id apparaît dans plusieurs fichiers, on garde une seule entrée.
- Le human_label non vide gagne (un label déjà annoté n'est pas écrasé par un vide).
- La note non vide gagne aussi.
- Les champs paper_title, abstract, paper_id sont conservés (ils viennent du premier fichier qui les a).
"""
import argparse
import json


def merge(paths: list[str], out: str) -> None:
    merged: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for a in data["annotations"]:
            cid = a["claim_id"]
            if cid not in merged:
                merged[cid] = dict(a)
                continue
            # fusion : garde le label non vide, la note non vide, et les champs manquants
            cur = merged[cid]
            if not cur.get("human_label") and a.get("human_label"):
                cur["human_label"] = a["human_label"]
            if not cur.get("note") and a.get("note"):
                cur["note"] = a["note"]
            for field in ("paper_id", "paper_title", "abstract", "claim"):
                if not cur.get(field) and a.get(field):
                    cur[field] = a[field]
    result = {"claims_file": paths[0], "annotations": list(merged.values())}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    labelled = sum(1 for a in merged.values() if a.get("human_label"))
    print(f"{len(merged)} affirmations fusionnées ({labelled} labellisées) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="fichiers gold à fusionner (ordre = priorité)")
    ap.add_argument("--out", required=True, help="fichier de sortie")
    a = ap.parse_args()
    merge(a.inputs, a.out)


if __name__ == "__main__":
    main()