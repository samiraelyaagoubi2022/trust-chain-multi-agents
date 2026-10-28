"""Mesure le vérificateur contre l'annotation humaine.

    python scripts/evaluate_verifier.py --gold eval/gold_strong.json --verdicts data/verdicts_strong.json
    # comparer deux configurations (ex. avec / sans contrôle de citation) :
    python scripts/evaluate_verifier.py --gold G --verdicts A.json --compare B.json

Métriques : accuracy 3 classes, matrice de confusion, précision/rappel par classe,
kappa de Cohen, et détection d'erreurs (positif = affirmation NON pleinement étayée).

v2 : les verdicts "erreur" (appel LLM échoué) sont EXCLUS du calcul et comptés à part ;
     avertissement si rédacteur et vérificateur sont de la même famille de modèles.
"""
import argparse
import json

LABELS = ["etayee", "partielle", "non_etayee"]


def family(model: str) -> str:
    """Famille = fournisseur (partie avant '/'), ex. 'openai/gpt-oss-120b' -> 'openai'."""
    return model.split("/")[0].lower() if "/" in model else model.lower()


def model_info(verdicts_path: str) -> tuple[str, str]:
    with open(verdicts_path, encoding="utf-8") as f:
        v = json.load(f)
    verifier, writer = v.get("verifier_model", "?"), "?"
    try:
        with open(v["claims_file"], encoding="utf-8") as f:
            writer = json.load(f).get("writer_model", "?")
    except (OSError, KeyError, ValueError):
        pass
    return writer, verifier


def load_pairs(gold_path: str, verdicts_path: str):
    """Renvoie (paires, sans_label, verdicts_en_erreur, sans_verdict)."""
    with open(gold_path, encoding="utf-8") as f:
        gold = {a["claim_id"]: a for a in json.load(f)["annotations"]}
    with open(verdicts_path, encoding="utf-8") as f:
        verd = {v["claim_id"]: v for v in json.load(f)["verdicts"]}
    pairs, skipped, errors, missing = [], [], [], []
    for cid, a in gold.items():
        h = a.get("human_label", "").strip()
        if h not in LABELS:
            skipped.append(cid)
        elif cid not in verd:
            missing.append(cid)
        elif verd[cid]["label"] not in LABELS:      # "erreur" : ne compte ni pour ni contre
            errors.append(cid)
        else:
            pairs.append((cid, h, verd[cid]["label"], a["claim"], verd[cid].get("reason", "")))
    return pairs, skipped, errors, missing


def kappa(pairs) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(h == p for _, h, p, *_ in pairs) / n
    pe = sum((sum(h == c for _, h, *_ in pairs) / n) * (sum(p == c for _, _, p, *_ in pairs) / n)
             for c in LABELS)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def report(pairs, skipped, title="", errors=(), missing=(), models=None):
    n = len(pairs)
    print(f"=== {title} ===  ({n} affirmations comparées, {len(skipped)} sans label)")
    if models:
        w, v = models
        print(f"Rédacteur : {w}   |   Vérificateur : {v}")
        if w not in ("?", "synthetic") and v != "?" and family(w) == family(v):
            print("ATTENTION : rédacteur et vérificateur sont de la même famille de modèles ; "
                  "leurs erreurs sont corrélées et le score est probablement optimiste.")
    if errors:
        print(f"{len(errors)} verdict(s) en ERREUR exclus du calcul : {', '.join(errors)} -> relancez le vérificateur")
    if missing:
        print(f"{len(missing)} affirmation(s) annotée(s) sans verdict : {', '.join(missing)}")
    if not n:
        return {}
    acc = sum(h == p for _, h, p, *_ in pairs) / n
    print(f"Accuracy 3 classes : {acc:.0%}   |   kappa de Cohen : {kappa(pairs):.2f}")

    print("\nMatrice de confusion (lignes = humain, colonnes = vérificateur)")
    print(f"{'':12}" + "".join(f"{c:>12}" for c in LABELS))
    for h in LABELS:
        print(f"{h:12}" + "".join(f"{sum(1 for _, hh, pp, *_ in pairs if hh == h and pp == c):>12}" for c in LABELS))

    print("\nPar classe (précision / rappel / F1)")
    for c in LABELS:
        tp = sum(1 for _, h, p, *_ in pairs if h == c and p == c)
        fp = sum(1 for _, h, p, *_ in pairs if h != c and p == c)
        fn = sum(1 for _, h, p, *_ in pairs if h == c and p != c)
        p_, r_, f_ = prf(tp, fp, fn)
        print(f"  {c:11} P={p_:.2f}  R={r_:.2f}  F1={f_:.2f}   (support humain: {tp + fn})")

    # détection d'erreurs : positif = pas pleinement étayée
    tp = sum(1 for _, h, p, *_ in pairs if h != "etayee" and p != "etayee")
    fp = sum(1 for _, h, p, *_ in pairs if h == "etayee" and p != "etayee")
    fn = sum(1 for _, h, p, *_ in pairs if h != "etayee" and p == "etayee")
    ep, er, ef = prf(tp, fp, fn)
    print(f"\nDétection d'erreurs (positif = partielle ou non étayée) : "
          f"P={ep:.2f}  R={er:.2f}  F1={ef:.2f}   (vraies erreurs: {tp + fn}, manquées: {fn}, fausses alertes: {fp})")
    if tp + fn < 5:
        print(f"Peu de vraies erreurs ({tp + fn}) : le rappel est peu fiable. "
              f"Ajoutez des erreurs synthétiques (scripts/make_synthetic_errors.py).")

    wrong = [x for x in pairs if x[1] != x[2]]
    if wrong:
        print("\nDésaccords :")
        for cid, h, p, claim, reason in wrong:
            print(f"  {cid}: humain={h} / vérificateur={p}\n     claim : {claim}\n     raison: {reason}")
    return {"accuracy": acc, "kappa": kappa(pairs), "err_precision": ep, "err_recall": er, "err_f1": ef}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--compare", help="second fichier de verdicts à comparer")
    a = ap.parse_args()
    pairs, skipped, errors, missing = load_pairs(a.gold, a.verdicts)
    r1 = report(pairs, skipped, a.verdicts, errors, missing, model_info(a.verdicts))
    if a.compare:
        print()
        p2, s2, e2, m2 = load_pairs(a.gold, a.compare)
        r2 = report(p2, s2, a.compare, e2, m2, model_info(a.compare))
        if r1 and r2:
            print(f"\nDifférence d'accuracy : {r2['accuracy'] - r1['accuracy']:+.0%} | "
                  f"de rappel d'erreurs : {r2['err_recall'] - r1['err_recall']:+.2f}")


if __name__ == "__main__":
    main()