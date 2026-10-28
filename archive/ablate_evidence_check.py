"""Ablation du contrôle de citation SANS nouvel appel à Groq (issue #8).

Le vérificateur enregistre, pour chaque verdict, s'il a été rétrogradé par le contrôle de citation
("[downgraded ..." dans la raison). On peut donc recalculer le verdict qu'aurait donné le LLM seul.

    python scripts/ablate_evidence_check.py --gold eval/gold_samefamily.json \
        eval/results/verdicts_samefamily_v0.json eval/results/verdicts_samefamily_v3.json
"""
import argparse
import contextlib
import json
import sys

LABELS = ("etayee", "partielle", "non_etayee")


def label_without_check(v: dict) -> str:
    """Le verdict du LLM seul : on annule une rétrogradation faite par le contrôle mécanique."""
    if v["label"] == "partielle" and str(v.get("reason", "")).startswith("[downgraded"):
        return "etayee"
    return v["label"]


def metrics(gold: dict, labels: dict) -> dict:
    ids = [i for i in gold if gold[i] in LABELS and labels.get(i) in LABELS]
    errors = [i for i in ids if gold[i] != "etayee"]
    good = [i for i in ids if gold[i] == "etayee"]
    return {"n": len(ids), "false_alarms": sum(labels[i] != "etayee" for i in good), "n_good": len(good),
            "detected": sum(labels[i] != "etayee" for i in errors), "n_err": len(errors),
            "accuracy": sum(labels[i] == gold[i] for i in ids) / len(ids) if ids else float("nan")}


def downgraded(verdicts: list[dict], gold: dict) -> list[tuple]:
    """Verdicts downgraded by the mechanical check: (claim id, human label, reason)."""
    return [(v["claim_id"], gold.get(v["claim_id"], "?"), v.get("reason", ""))
            for v in verdicts if label_without_check(v) != v["label"]]


def one_line(text: str, width: int = 110) -> str:
    """Single line, no CR/LF or odd spacing: keeps the table readable in a Windows console."""
    return " ".join(str(text).split())[:width]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", help="écrit le rapport dans ce fichier (UTF-8) au lieu de l'écran")
    ap.add_argument("--list", action="store_true",
                    help="liste les verdicts rétrogradés avec le label humain (fausse alerte ou vraie erreur ?)")
    ap.add_argument("verdicts", nargs="+")
    a = ap.parse_args()
    if a.out:                                   # le fichier est écrit en UTF-8 par Python lui-même
        with open(a.out, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            report(a)
        print(f"Rapport écrit dans {a.out} (ouvrez-le avec : Get-Content -Encoding utf8 {a.out})")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        report(a)


def report(a):
    with open(a.gold, encoding="utf-8") as f:
        gold = {x["claim_id"]: (x.get("human_label") or "").strip() for x in json.load(f)["annotations"]}
    print(f"{'fichier':48}{'rétrogradés':>12} | {'AVEC contrôle':^30} | {'SANS contrôle':^30}")
    print(f"{'':48}{'':>12} | {'fausses al.':>12}{'détectées':>10}{'acc.':>8} | {'fausses al.':>12}{'détectées':>10}{'acc.':>8}")
    for path in a.verdicts:
        with open(path, encoding="utf-8") as f:
            vs = json.load(f)["verdicts"]
        with_c = {v["claim_id"]: v["label"] for v in vs}
        without = {v["claim_id"]: label_without_check(v) for v in vs}
        down = sum(1 for v in vs if label_without_check(v) != v["label"])
        m1, m2 = metrics(gold, with_c), metrics(gold, without)
        fa = lambda m: f"{m['false_alarms']}/{m['n_good']}"
        det = lambda m: f"{m['detected']}/{m['n_err']}"
        print(f"{path[-47:]:48}{down:>12} | {fa(m1):>12}{det(m1):>10}{m1['accuracy']:>8.0%} | "
              f"{fa(m2):>12}{det(m2):>10}{m2['accuracy']:>8.0%}")
        if a.list:
            for cid, human, reason in downgraded(vs, gold):
                kind = "FAUSSE ALERTE (bonne affirmation)" if human == "etayee" else f"vraie erreur ({human})"
                print(f"      {cid:5} {kind:36} {one_line(reason)}")


if __name__ == "__main__":
    main()