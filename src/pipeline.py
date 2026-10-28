"""J4 - Pipeline (Orchestrateur) : Gère le flux entre les 3 Agents (Extraction, Vérification, Synthèse).
Calcule également les métriques de performance (temps et tokens) attribuées à chaque agent.

  writer_only  the writer's claims, untouched                      (baseline 1)
  self_review  the writer re-reads its own claims, same context    (baseline 2)
  verified     independent verifier -> writer corrects or deletes the rejected claims
               -> only the re-verified ones are re-checked, at most MAX_ROUNDS rounds.
               What is still rejected after the last round is dropped. Only claims judged
               "etayee" by the verifier reach the final note.

Run from the project root, on an existing claims file (papers + claims):
    python -m src.pipeline --claims eval/claims_strong.json --mode verified --prompt v3 \
        --out eval/results/pipeline_verified.json --note eval/results/note_verified.md
Without --claims, the writer first drafts claims from data/papers.json (needs --topic).
"""
import argparse
import json
import os
import sys
import time

from src import llm
from src.agents import verifier, writer
from src.config import CLAIMS_PATH, MAX_ROUNDS, MODEL, PAPERS_PATH, VERIFIER_MODEL
from src.schemas import Claim, Paper, Verdict

MODES = ("writer_only", "self_review", "verified")
BAD = ("partielle", "non_etayee")


def load_verdict_cache(paths: list[str]) -> dict:
    """Verdicts already obtained in earlier runs, keyed by (claim text, paper, prompt version, model)."""
    cache: dict = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            vf = json.load(f)
        if not vf.get("prompt_version") or not vf.get("verifier_model"):
            print(f"[cache] ignoré (version de consigne ou modèle inconnus) : {path}", file=sys.stderr)
            continue
        with open(vf["claims_file"], encoding="utf-8") as f:
            by_id = {c["id"]: c for c in json.load(f)["claims"]}
        for d in vf["verdicts"]:
            c = by_id.get(d["claim_id"])
            if c and d["label"] != "erreur":
                cache[(c["text"], c["paper_id"], vf["prompt_version"], vf["verifier_model"])] = d
    return cache


def load_checkpoint(path: str) -> dict:
    """Verdicts saved during an interrupted run of the same output file."""
    try:
        with open(path, encoding="utf-8") as f:
            return {tuple(e["key"]): e["verdict"] for e in json.load(f)}
    except (OSError, ValueError, KeyError):
        return {}


def save_checkpoint(path: str, entries: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"key": list(k), "verdict": v} for k, v in entries.items()], f, ensure_ascii=False)


def apply_changes(claims: list[Claim], changes: dict[str, str | None]):
    """changes = {id: new text, or None to delete}. Returns (claims, corrected ids, removed ids)."""
    out, corrected, removed = [], [], []
    for c in claims:
        if c.id in changes:
            new = changes[c.id]
            if new is None:
                removed.append(c.id)
                continue
            if new != c.text:
                corrected.append(c.id)
                c = Claim(id=c.id, text=new, paper_id=c.paper_id)
        out.append(c)
    return out, corrected, removed


def run_mode(mode: str, claims: list[Claim], papers: list[Paper], *, max_rounds: int = MAX_ROUNDS,
             prompt: str = "v3", check_evidence: bool = True, writer_model: str | None = None,
             verifier_model: str | None = None, progress: bool = False, cache: dict | None = None,
             on_fresh=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode inconnu : {mode}")
    llm.reset_stats()
    current = list(claims)
    corrected: set[str] = set()
    removed: dict[str, str] = {}            # id -> why
    verdicts: dict[str, Verdict] = {}
    rounds: list[dict] = []

    if mode == "self_review":
        current, corr, rem = apply_changes(current, writer.self_review(current, papers, writer_model))
        corrected.update(corr)
        removed.update({i: "self_review" for i in rem})

    elif mode == "verified":
        for rnd in range(1, max_rounds + 2):          # last round only re-checks, never revises
            todo = [c for c in current if c.id not in verdicts]
            fresh, reused = [], 0
            for c in todo:
                key = (c.text, c.paper_id, prompt, verifier_model or VERIFIER_MODEL)
                if cache and key in cache:
                    verdicts[c.id] = Verdict(**{**cache[key], "claim_id": c.id})
                    reused += 1
                else:
                    fresh.append(c)
            if fresh:
                fresh_by_id = {c.id: c for c in fresh}
                key_of = lambda c: (c.text, c.paper_id, prompt, verifier_model or VERIFIER_MODEL)
                notify = (lambda v: on_fresh(key_of(fresh_by_id[v.claim_id]), v.model_dump())) if on_fresh else None
                for v in verifier.verify_all(fresh, papers, progress=progress, check_evidence=check_evidence,
                                             model=verifier_model, prompt=prompt, on_verdict=notify):
                    verdicts[v.claim_id] = v
            bad = [c for c in current if verdicts[c.id].label in BAD]
            rounds.append({"round": rnd, "verified": len(fresh), "reused": reused,
                           "rejected": [c.id for c in bad]})
            if not bad:
                break
            if rnd > max_rounds:                       # budget spent: drop what still fails
                removed.update({c.id: "still_rejected" for c in bad})
                current = [c for c in current if c.id not in removed]
                break
            changes = writer.revise([(c, verdicts[c.id].reason) for c in bad], papers, model=writer_model)
            current, corr, rem = apply_changes(current, changes)
            corrected.update(corr)
            removed.update({i: "deleted_by_writer" for i in rem})
            for cid in corr + rem:                     # changed claims must be verified again
                verdicts.pop(cid, None)

    unverified = [c.id for c in current if mode == "verified" and verdicts[c.id].label not in BAD + ("etayee",)]
    final = [c for c in current if mode != "verified" or verdicts[c.id].label == "etayee"]
    return {
        "mode": mode, "claims_final": [c.model_dump() for c in final],
        "corrected": sorted(corrected - set(removed)), "removed": removed, "unverified": unverified,
        "rounds": rounds, "verdicts": {k: v.model_dump() for k, v in verdicts.items()},
        "stats": dict(llm.STATS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=None, help="fichier de claims existant (articles + affirmations)")
    ap.add_argument("--topic", default="", help="sujet (obligatoire sans --claims)")
    ap.add_argument("--papers", default=PAPERS_PATH)
    ap.add_argument("--mode", choices=MODES, default="verified")
    ap.add_argument("--prompt", choices=sorted(verifier.PROMPTS), default="v3")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--writer-model", default=None)
    ap.add_argument("--verifier-model", default=None)
    ap.add_argument("--wait-quota", dest="wait_quota", action="store_true")
    ap.add_argument("--reuse-verdicts", nargs="*", default=[], metavar="FICHIER",
                    help="fichiers de verdicts déjà obtenus (même consigne, même modèle) : évite de les refaire")
    ap.add_argument("--out", default="data/pipeline.json")
    ap.add_argument("--note", default=None, help="écrit la note finale en Markdown via l'Agent 3 (LLM)")
    a = ap.parse_args()

    llm.WAIT_FOR_QUOTA = a.wait_quota

    # =========================================================================
    # INITIALISATION DES MÉTRIQUES MULTI-AGENTS
    # =========================================================================
    metrics = {
        "Agent_3_Extraction": {"tokens": 0, "seconds": 0.0, "claims_extraits": 0},
        "Agent_2_Auditeur":   {"tokens": 0, "seconds": 0.0, "claims_verifies": 0, "claims_corriges": 0},
        "Agent_3_Synthese":   {"tokens": 0, "seconds": 0.0, "mots_generes": 0}
    }

    # =========================================================================
    # AGENT 3 : EXTRACTION (Si on ne charge pas un fichier existant)
    # =========================================================================
    start_t = time.time()
    llm.reset_stats()
    if a.claims:
        with open(a.claims, encoding="utf-8") as f:
            data = json.load(f)
        papers = [Paper(**p) for p in data["papers"]]
        claims = [Claim(**c) for c in data["claims"]]
        topic = data.get("topic", a.topic)
        metrics["Agent_3_Extraction"]["claims_extraits"] = len(claims)
    else:
        if not a.topic:
            sys.exit("--topic est obligatoire sans --claims")
        papers = writer.load_papers(a.papers)
        claims = writer.write_note(a.topic, papers, model=a.writer_model)
        topic = a.topic
        
        metrics["Agent_3_Extraction"]["claims_extraits"] = len(claims)
        metrics["Agent_3_Extraction"]["tokens"] = llm.STATS['prompt_tokens'] + llm.STATS['completion_tokens']
        metrics["Agent_3_Extraction"]["seconds"] = round(time.time() - start_t, 1)
        print(f"[AGENT] Agent 3 (Extraction) : {len(claims)} claims extraits en {metrics['Agent_3_Extraction']['seconds']}s.")

    # =========================================================================
    # AGENT 2 : AUDITEUR (Vérification + Correction)
    # =========================================================================
    start_t = time.time()
    llm.reset_stats()
    
    ckpt_path = a.out + ".checkpoint.json"          # un arrêt ne fait plus perdre les verdicts déjà payés
    cache = load_verdict_cache(a.reuse_verdicts)
    saved = load_checkpoint(ckpt_path)
    if saved:
        print(f"Reprise : {len(saved)} verdicts retrouvés dans {ckpt_path}", file=sys.stderr)
    cache.update(saved)

    def remember(key, verdict):
        saved[key] = verdict
        save_checkpoint(ckpt_path, saved)

    try:
        res = run_mode(a.mode, claims, papers, max_rounds=a.max_rounds, prompt=a.prompt,
                       writer_model=a.writer_model, verifier_model=a.verifier_model, progress=True,
                       cache=cache, on_fresh=remember)
    except RuntimeError as e:
        sys.exit(f"\nARRET : {e}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        
    metrics["Agent_2_Auditeur"]["tokens"] = llm.STATS['prompt_tokens'] + llm.STATS['completion_tokens']
    metrics["Agent_2_Auditeur"]["seconds"] = round(time.time() - start_t, 1)
    metrics["Agent_2_Auditeur"]["claims_verifies"] = len(claims)
    metrics["Agent_2_Auditeur"]["claims_corriges"] = len(res['corrected'])
    print(f"[AGENT] Agent 2 (Auditeur) : {len(claims)} vérifiés, {len(res['corrected'])} corrigés en {metrics['Agent_2_Auditeur']['seconds']}s.")

    # Sauvegarde intermédiaire du JSON avec les métriques des agents 3 et 2
    res.update({
        "topic": topic, "claims_file": a.claims, "prompt": a.prompt, "max_rounds": a.max_rounds,
        "writer_model": a.writer_model or MODEL, "verifier_model": a.verifier_model or VERIFIER_MODEL,
        "n_input": len(claims), "metrics": metrics
    })
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # AGENT 3 : SYNTHÈSE FINALE (Appel LLM)
    # =========================================================================
    if a.note:
        print("\n[AGENT] Agent 3 (Synthèse) : Rédaction de la note finale...")
        start_t = time.time()
        llm.reset_stats()
        
        final_claims = [Claim(**c) for c in res["claims_final"]]
        
        # Utilisation de la nouvelle fonction de writer.py
        synthesis_text = writer.synthesize(topic, final_claims, model=a.writer_model)
        
        metrics["Agent_3_Synthese"]["tokens"] = llm.STATS['prompt_tokens'] + llm.STATS['completion_tokens']
        metrics["Agent_3_Synthese"]["seconds"] = round(time.time() - start_t, 1)
        metrics["Agent_3_Synthese"]["mots_generes"] = len(synthesis_text.split())
        
        # Mise à jour finale du JSON avec les métriques complètes
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
            
        with open(a.note, "w", encoding="utf-8") as f:
            f.write(synthesis_text)
            
        print(f"[OK] Synthèse générée ({metrics['Agent_3_Synthese']['mots_generes']} mots) -> {a.note}")

    # =========================================================================
    # AFFICHAGE DU RAPPORT D'ATTRIBUTION DES TÂCHES
    # =========================================================================
    s = res["stats"]
    print(f"\nmode={a.mode} | {len(claims)} affirmations -> {len(res['claims_final'])} finales "
          f"| corrigées {len(res['corrected'])} | supprimées {len(res['removed'])} "
          f"| non vérifiées {len(res['unverified'])} | tours {len(res['rounds'])} "
          f"| Total : {s['prompt_tokens']+s['completion_tokens']} tokens | {s['seconds']:.1f}s\n")
          
    print("="*50)
    print("[METRICS] RÉPARTITION DES TÂCHES PAR AGENT")
    print("="*50)
    for agent, m in metrics.items():
        print(f"[{agent}]")
        for k, v in m.items():
            print(f"  - {k} : {v}")
        print()


if __name__ == "__main__":
    main()