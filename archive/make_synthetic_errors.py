"""Fabrique des affirmations FAUSSES de façon déterministe (sans LLM) pour tester le vérificateur.

Trois types d'erreur, avec le label attendu connu par construction :
  number_changed     un chiffre est modifié                      -> non_etayee
  direction_reversed un mot de sens est inversé (outperforms...) -> non_etayee
  unsupported_cause  un ajout plausible mais inventé (chiffre, étude, cause) -> partielle

Usage :
    python scripts/make_synthetic_errors.py --claims eval/claims_strong.json \
        --gold eval/gold_strong.json --per-type 3

--gold (recommandé) : on ne part que des affirmations que VOUS avez annotées "etayee",
sinon la modification pourrait tomber sur une affirmation déjà fausse.

Sorties : eval/claims_synthetic.json (pour le vérificateur) et eval/gold_synthetic.json
(labels pré-remplis). Ces résultats se rapportent SÉPARÉMENT des vraies erreurs du rédacteur :
des erreurs fabriquées sont plus faciles à détecter que des erreurs naturelles.
"""
import argparse
import json
import random
import re

NUM = re.compile(r"(?<![\w@.\-^,])(\d+(?:\.\d+)?)(\s?%)?(?!\w|,\d)")

REVERSALS = {
    "outperforms": "underperforms", "outperform": "underperform", "outperformed": "underperformed",
    "improves": "degrades", "improve": "degrade",
    "higher": "lower", "highest": "lowest", "lower": "higher", "lowest": "highest",
    "best": "worst", "faster": "slower", "slower": "faster",
    "increases": "decreases", "increase": "decrease", "reduces": "increases", "reduce": "increase",
    "better": "worse", "more": "less", "less": "more",
}
REV = re.compile(r"\b(" + "|".join(sorted(REVERSALS, key=len, reverse=True)) + r")\b", re.IGNORECASE)

CAUSES = [
    ", outperforming the strongest baseline by 12%",
    ", as confirmed by a user study with 200 participants",
    ", and these results were replicated on three additional datasets",
    ", which the authors attribute to the use of a larger model",
]

EXPECTED = {"number_changed": "non_etayee", "direction_reversed": "non_etayee",
            "unsupported_cause": "partielle"}


def _bump(num: str, pct: bool) -> str:
    if "." in num:
        decimals = len(num.split(".")[1])
        v = float(num)
        v2 = v + (0.1 if v < 1 else 1.0)          # 0.66 -> 0.76 ; 4.55 -> 5.55
        if v < 1 and v2 >= 1:
            v2 = v - 0.1
        return f"{v2:.{decimals}f}"
    n = int(num)
    if pct:
        return str(n - 15 if n > 50 else n + 15)  # 20% -> 35% ; 91% -> 76%
    return str(round(n * 1.5))                    # 866 -> 1299


def perturb_number(text: str, rng: random.Random):
    """Change one number. Returns (new_text, old, new) or None if no suitable number."""
    cands = [m for m in NUM.finditer(text)
             if "." in m.group(1) or m.group(2) or int(m.group(1)) >= 10]
    if not cands:
        return None
    m = rng.choice(cands)
    old = m.group(1)
    new = _bump(old, bool(m.group(2)))
    return text[:m.start(1)] + new + text[m.end(1):], old, new


def perturb_direction(text: str):
    """Reverse the first direction word found. Returns (new_text, old, new) or None."""
    m = REV.search(text)
    if not m:
        return None
    old = m.group(1)
    new = REVERSALS[old.lower()]
    if old[0].isupper():
        new = new.capitalize()
    return text[:m.start(1)] + new + text[m.end(1):], old, new


def add_unsupported_cause(text: str, rng: random.Random):
    clause = rng.choice(CAUSES)
    return text.rstrip(" .") + clause + ".", "", clause.strip(", ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--gold", default=None, help="gold annoté : ne garde que les affirmations 'etayee'")
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-claims", default="eval/claims_synthetic.json")
    ap.add_argument("--out-gold", default="eval/gold_synthetic.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    data = json.load(open(a.claims, encoding="utf-8"))
    papers = {p["id"]: p for p in data["papers"]}
    base = data["claims"]
    if a.gold:
        ok = {x["claim_id"] for x in json.load(open(a.gold, encoding="utf-8"))["annotations"]
              if x.get("human_label", "").strip() == "etayee"}
        base = [c for c in base if c["id"] in ok]
        print(f"{len(base)} affirmations annotées 'etayee' utilisables comme base")

    makers = {
        "number_changed": lambda t: perturb_number(t, rng),
        "direction_reversed": perturb_direction,
        "unsupported_cause": lambda t: add_unsupported_cause(t, rng),
    }
    used, synth, meta, annotations = set(), [], [], []
    for kind, fn in makers.items():
        pool = [c for c in base if c["id"] not in used]
        rng.shuffle(pool)
        made = 0
        for c in pool:
            if made >= a.per_type:
                break
            res = fn(c["text"])
            if not res:
                continue
            new_text, old, new = res
            sid = f"S{len(synth) + 1}"
            synth.append({"id": sid, "text": new_text, "paper_id": c["paper_id"]})
            meta.append({"id": sid, "base_claim_id": c["id"], "kind": kind, "original": c["text"]})
            p = papers[c["paper_id"]]
            annotations.append({"claim_id": sid, "paper_id": c["paper_id"], "claim": new_text,
                                "paper_title": p["title"], "abstract": p["abstract"],
                                "human_label": EXPECTED[kind], "note": f"synthetic:{kind}",
                                "synthetic": True})
            used.add(c["id"])
            made += 1
            print(f"{sid} [{kind}] {c['id']}: {old!r} -> {new!r}\n     {new_text}")
        if made < a.per_type:
            print(f"(seulement {made}/{a.per_type} affirmations adaptées pour {kind})")

    json.dump({"topic": data.get("topic", ""), "writer_model": "synthetic",
               "papers": data["papers"], "claims": synth, "synthetic": meta},
              open(a.out_claims, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"claims_file": a.out_claims, "annotations": annotations},
              open(a.out_gold, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{len(synth)} affirmations synthétiques -> {a.out_claims} et {a.out_gold}")


if __name__ == "__main__":
    main()