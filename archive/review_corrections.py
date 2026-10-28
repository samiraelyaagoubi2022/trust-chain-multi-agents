"""Relecture HUMAINE des affirmations réécrites par un mode du pipeline (issue #7).

    python scripts/review_corrections.py --evalset eval/claims_evalset.json \
        --result eval/results/pipe_verified.json --out eval/results/review_verified.json

Pour chaque affirmation réécrite (erreur corrigée à relire, ou bonne affirmation modifiée), vous voyez le
résumé, l'affirmation d'origine et le nouveau texte, puis vous répondez e / p / n (mêmes définitions que
pour l'annotation). Sauvegarde après chaque réponse ; relancer la commande reprend où vous en étiez.
A la fin : le nombre total d'erreurs réellement réparées et de bonnes affirmations abîmées.
"""
import argparse
import json
import os
import sys
import textwrap

try:
    from scripts.evaluate_pipeline import evaluate
except ModuleNotFoundError:               # lancé comme script : le dossier scripts/ est sur le chemin
    from evaluate_pipeline import evaluate

KEYS = {"e": "etayee", "p": "partielle", "n": "non_etayee"}


def summarize(r: dict, reviews: dict) -> dict:
    """r = sortie de evaluate(), reviews = {id: {"label": ...}}."""
    lab = lambda i: reviews.get(i, {}).get("label")
    fixed = [i for i in r["err_review"] if lab(i) == "etayee"]
    still_wrong = [i for i in r["err_review"] if lab(i) in ("partielle", "non_etayee")]
    altered_ok = [i for i in r["good_altered"] if lab(i) == "etayee"]
    altered_bad = [i for i in r["good_altered"] if lab(i) in ("partielle", "non_etayee")]
    pending = [i for i in r["err_review"] + r["good_altered"] if not lab(i)]
    return {"errors_repaired": len(r["err_repaired"]) + len(fixed), "errors_repaired_auto": len(r["err_repaired"]),
            "errors_repaired_by_hand_review": len(fixed), "rewrites_still_wrong": still_wrong,
            "good_altered_ok": altered_ok, "good_altered_damaged": altered_bad, "pending": pending}


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.evalset, encoding="utf-8") as f:
        evalset = json.load(f)
    with open(a.result, encoding="utf-8") as f:
        result = json.load(f)
    r = evaluate(evalset, result)
    papers = {p["id"]: p for p in evalset["papers"]}
    claims = {c["id"]: c for c in evalset["claims"]}
    data = {"result": a.result, "reviews": {}}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            data = json.load(f)
    reviews = data["reviews"]
    todo = [i for i in r["err_review"] + r["good_altered"] if i not in reviews]
    print(f"{len(r['err_review'])} corrections d'erreurs et {len(r['good_altered'])} bonnes affirmations "
          f"modifiées à relire ; {len(todo)} restantes.")
    for n, cid in enumerate(todo, 1):
        p = papers[claims[cid]["paper_id"]]
        kind = "ERREUR corrigée" if cid in r["err_review"] else "BONNE affirmation modifiée"
        print("\n" + "=" * 100 + f"\n[{n}/{len(todo)}] {cid} ({kind}) - {p['title']}\n" + "-" * 100)
        print(textwrap.fill(p["abstract"], 100) + "\n" + "-" * 100)
        print("AVANT : " + textwrap.fill(claims[cid]["text"], 100, subsequent_indent="        "))
        print("APRES : " + textwrap.fill(r["final_texts"][cid], 100, subsequent_indent="        "))
        while True:
            ans = input("\nLe NOUVEAU texte est-il fidèle au résumé ? [e]tayee [p]artielle [n]on_etayee [s]kip [q]uit > ").strip().lower()
            if ans in KEYS:
                note = input("note (Entrée pour passer) > ").strip() if ans != "e" else ""
                reviews[cid] = {"label": KEYS[ans], "note": note, "before": claims[cid]["text"],
                                "after": r["final_texts"][cid]}
                _save(a.out, data)
                break
            if ans == "s":
                break
            if ans == "q":
                _save(a.out, data)
                return
    s = summarize(r, reviews)
    n_err = sum(1 for m in evalset["meta"].values() if m["gold"] != "etayee")
    print(f"\nErreurs réellement réparées : {s['errors_repaired']}/{n_err} "
          f"({s['errors_repaired_auto']} automatiques + {s['errors_repaired_by_hand_review']} après votre relecture)")
    print(f"Réécritures encore fausses : {s['rewrites_still_wrong'] or 'aucune'}")
    print(f"Bonnes affirmations abîmées par la réécriture : {s['good_altered_damaged'] or 'aucune'}")
    if s["pending"]:
        print(f"Non relues : {s['pending']}")


if __name__ == "__main__":
    main()