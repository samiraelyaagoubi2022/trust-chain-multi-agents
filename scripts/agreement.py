"""Accord entre deux annotateurs (issue #19) : accuracy, kappa de Cohen, désaccords.

Procédure : copiez eval/gold_samefamily.json, effacez les "human_label" de la copie, donnez-la à une autre
personne (python scripts/annotate.py copie.json --sample 15 --seed 1), puis :

    python scripts/agreement.py eval/gold_samefamily.json copie_annotee.json
"""
import json
import sys

LABELS = ("etayee", "partielle", "non_etayee")


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in LABELS)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare(g1: dict, g2: dict):
    """g = {claim_id: (label, text)}. Returns common ids, labels of both, disagreements."""
    ids = [i for i in g1 if g1[i][0] in LABELS and i in g2 and g2[i][0] in LABELS]
    a, b = [g1[i][0] for i in ids], [g2[i][0] for i in ids]
    return ids, a, b, [(i, g1[i][0], g2[i][0], g1[i][1]) for i in ids if g1[i][0] != g2[i][0]]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:   # <- utf-8-sig au lieu de utf-8
        return {x["claim_id"]: ((x.get("human_label") or "").strip(), x["claim"]) for x in json.load(f)["annotations"]}

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ids, a, b, diff = compare(_load(sys.argv[1]), _load(sys.argv[2]))
    print(f"{len(ids)} affirmations annotées par les deux personnes")
    if ids:
        print(f"Accord brut : {sum(x == y for x, y in zip(a, b)) / len(ids):.0%} | kappa de Cohen : {kappa(a, b):.2f}")
    for cid, l1, l2, text in diff:
        print(f"  {cid}: {l1} / {l2}\n     {text}")


if __name__ == "__main__":
    main()