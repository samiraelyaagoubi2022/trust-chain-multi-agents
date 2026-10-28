"""Extrait un sous-ensemble d'un gold original correspondant aux claim_id d'un fichier aveugle.

Usage :
    python scripts/extract_subset.py eval/gold_evalset.json eval/gold_blind_15.json eval/gold_original_15.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", help="gold original (avec labels)")
    ap.add_argument("subset", help="fichier aveugle (mêmes claim_id, labels vides)")
    ap.add_argument("out", help="fichier de sortie")
    a = ap.parse_args()

    with open(a.original, encoding="utf-8") as f:
        orig = json.load(f)
    with open(a.subset, encoding="utf-8") as f:
        sub = json.load(f)

    ids = {x["claim_id"] for x in sub["annotations"]}
    orig["annotations"] = [x for x in orig["annotations"] if x["claim_id"] in ids]

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(orig, f, ensure_ascii=False, indent=2)

    print(f"{len(orig['annotations'])} affirmations extraites -> {a.out}")
    print(f"Comparez avec : python scripts/agreement.py {a.out} {a.subset}")


if __name__ == "__main__":
    main()