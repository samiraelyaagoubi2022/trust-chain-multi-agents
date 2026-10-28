"""Annotation humaine en terminal : une affirmation à la fois, avec le résumé de l'article.

    python scripts/annotate.py eval/gold_strong.json
    python scripts/annotate.py eval/gold_samefamily.json --sample 15 --seed 1

Touches : e = etayee | p = partielle | n = non_etayee | s = passer | u = annuler la précédente | q = quitter
La progression est sauvegardée après chaque réponse : vous pouvez quitter et reprendre.

Définitions (identiques à celles données au vérificateur, pour mesurer des erreurs et non des
désaccords de vocabulaire) :
  etayee      : chaque élément de l'affirmation (chiffres, noms, sens, portée) est dit dans le résumé
  partielle   : une partie est dite, mais une autre manque, est exagérée, fusionnée avec autre chose,
                attribuée au mauvais objet, ou un conditionnel est devenu une certitude
  non_etayee  : le résumé ne le dit pas, ou dit le contraire (autre chiffre, résultat inversé)

A faire AVANT de lancer le vérificateur, pour ne pas être influencé.
"""
import argparse
import json
import os
import random
import re
import sys
import textwrap

KEYS = {"e": "etayee", "p": "partielle", "n": "non_etayee"}
NUM = re.compile(r"\d[\d.,]*\d|\d")


def _clean(text: str) -> str:
    return (text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-")
                .replace("\u2014", "-").replace("\u00a0", " "))


def missing_numbers(claim: str, abstract: str) -> list[str]:
    """Chiffres de l'affirmation introuvables dans le résumé (aide mécanique, pas un verdict)."""
    abs_clean = _clean(abstract)
    out = []
    for m in NUM.findall(_clean(claim)):
        n = m.rstrip(".,")
        if n and n not in abs_clean and n not in out:
            out.append(n)
    return out


def save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def show(item: dict, pos: int, total: int) -> None:
    print("\n" + "=" * 100)
    print(f"[{pos}/{total}]  {item['claim_id']}  -  {item['paper_title']}")
    print("-" * 100)
    print(textwrap.fill(_clean(item["abstract"]), width=100))
    print("-" * 100)
    print("AFFIRMATION : " + textwrap.fill(_clean(item["claim"]), width=100, subsequent_indent="              "))
    miss = missing_numbers(item["claim"], item["abstract"])
    if miss:
        print(f"ATTENTION (aide mécanique) : chiffres absents du résumé : {', '.join(miss)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--sample", type=int, default=0, help="n'annoter qu'un sous-ensemble aléatoire de N affirmations")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    with open(a.gold, encoding="utf-8") as f:
        data = json.load(f)
    items = data["annotations"]
    ids = [x["claim_id"] for x in items]
    if a.sample and a.sample < len(ids):
        keep = set(random.Random(a.seed).sample(ids, a.sample))
        items_todo = [x for x in items if x["claim_id"] in keep]
    else:
        items_todo = items
    total = len(items_todo)
    done_before = sum(1 for x in items_todo if x.get("human_label"))
    print(__doc__.split("A faire AVANT")[0])
    print(f"{total} affirmations à annoter dans ce fichier, {done_before} déjà faites.")

    history: list[dict] = []
    i = 0
    while i < len(items_todo):
        item = items_todo[i]
        if item.get("human_label"):
            i += 1
            continue
        show(item, i + 1, total)
        while True:
            ans = input("\n[e]tayee  [p]artielle  [n]on_etayee  [s]kip  [u]ndo  [q]uit > ").strip().lower()
            if ans in KEYS:
                item["human_label"] = KEYS[ans]
                if ans != "e":
                    item["note"] = input("note (pourquoi ? Entrée pour passer) > ").strip()
                history.append(item)
                save(a.gold, data)
                i += 1
                break
            if ans == "s":
                i += 1
                break
            if ans == "u":
                if not history:
                    print("Rien à annuler.")
                    continue
                last = history.pop()
                last["human_label"], last["note"] = "", ""
                save(a.gold, data)
                i = items_todo.index(last)
                break
            if ans == "q":
                save(a.gold, data)
                n = sum(1 for x in items_todo if x.get("human_label"))
                print(f"Sauvegardé : {n}/{total} annotées.")
                return
            print("Touche non reconnue.")
    n = sum(1 for x in items_todo if x.get("human_label"))
    print(f"\nTerminé : {n}/{total} annotées dans {a.gold}")


if __name__ == "__main__":
    main()