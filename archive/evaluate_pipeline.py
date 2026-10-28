"""Compare les modes du pipeline sur le jeu d'évaluation (labels connus).

    python scripts/evaluate_pipeline.py --evalset eval/claims_evalset.json \
        writer_only=eval/results/pipe_writer_only.json \
        self_review=eval/results/pipe_self_review.json \
        verified=eval/results/pipe_verified.json

Pour chaque mode :
  erreurs restantes     erreur présente, texte inchangé, dans la note finale   (le pire résultat)
  erreurs supprimées    erreur retirée de la note (supprimée ou non vérifiable)
  erreurs corrigées     le rédacteur a réécrit l'affirmation :
        réparées     le nouveau texte est (quasi) identique à l'affirmation d'origine  [erreurs synthétiques]
        à relire     à juger à la main (erreurs naturelles, ou réécriture différente)
  bonnes perdues        affirmation étayée supprimée à tort
  bonnes modifiées      affirmation étayée réécrite (risque d'introduire une erreur)
"""
import argparse
import json
import re
from difflib import SequenceMatcher


def _tokens(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", t.lower())


def _numbers(tokens: list[str]) -> list[str]:
    return sorted(t for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?%?", t))


def similar(a: str, b: str, threshold: float = 0.9) -> bool:
    """Same wording up to small differences, and EXACTLY the same numbers (0.76 is never 'close to' 0.66)."""
    ta, tb = _tokens(a), _tokens(b)
    if _numbers(ta) != _numbers(tb):
        return False
    return SequenceMatcher(None, ta, tb).ratio() >= threshold


def evaluate(evalset: dict, result: dict) -> dict:
    meta, texts = evalset["meta"], {c["id"]: c["text"] for c in evalset["claims"]}
    final = {c["id"]: c for c in result["claims_final"]}
    removed, unverified = set(result["removed"]), set(result["unverified"])
    out = {k: [] for k in ("err_remaining", "err_removed", "err_repaired", "err_review",
                           "good_kept", "good_lost", "good_altered")}
    for cid, m in meta.items():
        is_err = m["gold"] != "etayee"
        if cid in removed or cid in unverified:
            out["err_removed" if is_err else "good_lost"].append(cid)
        elif cid in final and final[cid]["text"] != texts[cid]:
            if not is_err:
                out["good_altered"].append(cid)
            elif m.get("original_text") and similar(final[cid]["text"], m["original_text"]):
                out["err_repaired"].append(cid)
            else:
                out["err_review"].append(cid)
        elif cid in final:
            out["err_remaining" if is_err else "good_kept"].append(cid)
    out["stats"] = result.get("stats", {})
    out["final_texts"] = {i: final[i]["text"] for i in out["err_review"] + out["good_altered"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("runs", nargs="+", help="mode=chemin_du_resultat.json")
    a = ap.parse_args()
    with open(a.evalset, encoding="utf-8") as f:
        evalset = json.load(f)
    n_err = sum(1 for m in evalset["meta"].values() if m["gold"] != "etayee")
    n_good = len(evalset["meta"]) - n_err
    print(f"Jeu : {len(evalset['meta'])} affirmations = {n_good} étayées + {n_err} avec erreur\n")
    head = f"{'mode':12}{'err. restantes':>16}{'err. supprimées':>17}{'err. réparées':>15}{'à relire':>10}" \
           f"{'bonnes perdues':>16}{'bonnes modifiées':>18}{'tokens':>10}"
    print(head)
    reviews = {}
    for run in a.runs:
        mode, path = run.split("=", 1)
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        r = evaluate(evalset, res)
        s = r["stats"]
        tok = s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
        print(f"{mode:12}{len(r['err_remaining']):>13}/{n_err:<3}{len(r['err_removed']):>14}/{n_err:<3}"
              f"{len(r['err_repaired']):>12}/{n_err:<3}{len(r['err_review']):>7}"
              f"{len(r['good_lost']):>13}/{n_good:<3}{len(r['good_altered']):>15}/{n_good:<3}{tok:>8}")
        reviews[mode] = r["final_texts"]
    for mode, items in reviews.items():
        if items:
            print(f"\nA relire à la main ({mode}) : le texte final est-il fidèle au résumé ?")
            for cid, text in items.items():
                print(f"  {cid}: {text}")


if __name__ == "__main__":
    main()