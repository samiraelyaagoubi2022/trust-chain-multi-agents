# scripts/corrector.py
"""Boucle de correction automatique : pour chaque claim rejeté, demander au writer
de le réécrire en se basant sur la justification du vérificateur, puis re-vérifier.

Usage :
    python scripts/corrector.py --claims eval/claims_merged.json --verdicts eval/results/verdicts_merged_v3.json --out eval/corrected_claims.json --wait-quota
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

# Ajouter le projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import llm
from src.config import VERIFIER_MODEL
from src.schemas import Claim, Paper, Verdict

load_dotenv(override=True)

CORRECTOR_PROMPT = """You are a claim corrector. You receive:
- A CLAIM that was judged "partial" or "unsupported" against a paper's abstract.
- The REASON why it was rejected.
- The TITLE and ABSTRACT of the paper.

Your task: rewrite the claim so that it becomes fully supported by the abstract.
Rules:
- Keep the claim's core message when possible.
- Remove any number not stated in the abstract.
- Remove any added content (dataset, baseline, study) not in the abstract.
- Fix wrong attributions (credit the right entity).
- Weaken modality when the abstract hedges ("suggest that" -> "suggests that").
- If the claim cannot be supported at all, return an empty string.

Answer only with JSON: {"corrected": "..."}"""


def correct_claim(claim_text: str, reason: str, paper: Paper) -> str:
    user = (
        f"CLAIM: {claim_text}\n\n"
        f"REASON FOR REJECTION: {reason}\n\n"
        f"TITLE: {paper.title}\n\n"
        f"ABSTRACT:\n{paper.abstract}"
    )
    try:
        out = llm.chat_json(CORRECTOR_PROMPT, user, temperature=0.0, model=VERIFIER_MODEL)
        return str(out.get("corrected", "")).strip()
    except Exception as e:
        print(f"  Erreur correction : {e}", file=sys.stderr)
        return ""


def verify_corrected(claim: Claim, paper: Paper) -> Verdict:
    from src.agents.verifier import verify_claim
    return verify_claim(claim, paper, prompt="v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-quota", dest="wait_quota", action="store_true",
                    help="si le quota Groq est épuisé, attendre et reprendre automatiquement")
    a = ap.parse_args()

    # Activer l'attente de quota (comme verifier.py et writer.py)
    llm.WAIT_FOR_QUOTA = a.wait_quota
    llm.reset_stats()

    # Charger les claims
    with open(a.claims, encoding="utf-8") as f:
        data = json.load(f)
    papers = {p["id"]: Paper(**p) for p in data["papers"]}
    claims = {c["id"]: Claim(**c) for c in data["claims"]}

    # Charger les verdicts
    with open(a.verdicts, encoding="utf-8") as f:
        verdicts_data = json.load(f)
    verdicts = {v["claim_id"]: Verdict(**v) for v in verdicts_data["verdicts"]}

    # Identifier les claims rejetés
    rejected = [(cid, v) for cid, v in verdicts.items() if v.label in ("partielle", "non_etayee")]
    print(f"{len(rejected)} claims rejetés à corriger")

    # Point de reprise
    partial = a.out + ".partial.json"
    done = {}
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            saved = json.load(f)
        done = {r["claim_id"]: r for r in saved.get("results", [])}
        print(f"Reprise : {len(done)} claims déjà traités")

    results = list(done.values())
    corrected_count = sum(1 for r in results if r.get("status") in ("fixed", "still_partial"))
    improved_count = sum(1 for r in results if r.get("status") == "fixed")

    def checkpoint():
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    for i, (cid, v) in enumerate(rejected, 1):
        if cid in done:
            continue

        claim = claims.get(cid)
        paper = papers.get(claim.paper_id) if claim else None
        if not claim or not paper:
            continue

        print(f"[{i}/{len(rejected)}] {cid} ({v.label})")

        corrected_text = correct_claim(claim.text, v.reason, paper)
        if not corrected_text:
            results.append({"claim_id": cid, "original": claim.text, "corrected": "",
                            "original_label": v.label, "new_label": None, "status": "unfixable"})
            checkpoint()
            continue

        corrected_count += 1
        corrected_claim = Claim(id=cid, text=corrected_text, paper_id=claim.paper_id)
        new_verdict = verify_corrected(corrected_claim, paper)
        print(f"  → nouveau verdict : {new_verdict.label}")

        status = "fixed" if new_verdict.label == "etayee" else "still_partial"
        if status == "fixed":
            improved_count += 1

        results.append({
            "claim_id": cid,
            "original": claim.text,
            "corrected": corrected_text,
            "original_label": v.label,
            "new_label": new_verdict.label,
            "new_reason": new_verdict.reason,
            "status": status,
        })
        checkpoint()

    # Écrire le résultat final
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    if os.path.exists(partial):
        os.remove(partial)

    print()
    print(f"=== RÉSULTAT ===")
    print(f"Claims rejetés : {len(rejected)}")
    print(f"Claims corrigés : {corrected_count}")
    print(f"Claims devenus 'etayee' : {improved_count}")
    if corrected_count > 0:
        print(f"Taux de correction réussie : {improved_count / corrected_count:.1%}")


if __name__ == "__main__":
    main()