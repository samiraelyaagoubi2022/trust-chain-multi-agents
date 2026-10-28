"""J3 - Verifier: checks each claim against the title + abstract of the cited paper ONLY.

Independence by construction: the prompt contains the claim and the title/abstract of
the cited paper. Nothing else - no other claims, no writer reasoning, no topic.
A different model than the writer is used (config.VERIFIER_MODEL).

v2: the title is part of the source (the human annotator saw it too), the evidence check
tolerates multi-sentence quotes ("...") and small formatting differences (but never a
different number), and --prompt v2 adds an explicit subject/modality/scope checklist.

v3: checklist refined so that (a) naming the authors' own framework instead of "we" is
FINE (fixes false positive F2), (b) only a STRONGER modality than the abstract is an
error (fixes false negative F27), and (c) a wrong number is ALWAYS "unsupported", never
"partial" (fixes X1, X3, Y1, Y2). Labels in the checklist are the English ones the
parser expects (supported / partial / unsupported).

Run from the project root:
    python -m src.agents.verifier --claims eval/claims_merged.json --prompt v3 \
        --out eval/results/verdicts_merged_v3.json
"""
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

from src import llm
from src.config import CLAIMS_PATH, VERDICTS_PATH, VERIFIER_MODEL
from src.schemas import Claim, Paper, Verdict

SYSTEM = """You are a strict fact-checker. You receive ONE claim and the TITLE and ABSTRACT of the paper it cites.
Decide whether the title and abstract support the claim. Use ONLY them. Ignore anything you know about the paper.

Labels:
- "supported": every part of the claim (numbers, names, direction of effects, scope) is stated in the abstract.
- "partial": part of the claim is stated, but another part is missing, exaggerated, merged from separate
  statements, attributed to the wrong thing, or turned from a hedge ("may", "would") into a certainty.
- "unsupported": the abstract does not state it, or contradicts it (different number, reversed result).

Check numbers digit by digit. Check causal words ("because", "by supplying"): the abstract must state the cause.

Answer only with this JSON:
{"verdict": "supported" | "partial" | "unsupported",
 "evidence": "<exact fragment(s) copied from the title/abstract; separate several fragments with ' ... '; empty if none>",
 "reason": "<one short sentence>"}"""

CHECKLIST = """

Before deciding, check each of these and mention the one that fails in "reason":
1. SUBJECT: who or what performs the action in the claim (the benchmark, the authors' method, a team, a metric)?
   Is it the same subject as in the abstract? A benchmark that "achieves" results or "proposes" a pipeline,
   when the abstract attributes this to the authors or their method, is "partial".
2. NUMBERS: every number, unit and comparison, digit by digit.
3. MODALITY: "can be interpreted to", "may", "aims to", "suggests" must not become "can", "does", "shows".
   A weaker statement turned into a stronger one is "partial".
4. SCOPE: words like all, always, consistently, any, only, most must be stated in the abstract.
5. ADDED CONTENT: any cause, baseline, dataset, study or number not in the title/abstract makes the claim at best "partial"."""

CHECKLIST_V3 = """

Before deciding, check each of these and mention the one that fails in "reason":
1. SUBJECT: who or what performs the action in the claim? It must be the entity the abstract credits with it.
   - FINE: the claim names the authors' own contribution when the abstract introduces that entity as such
     ("this paper introduces X", "we present X"): "X proposes ..." is equivalent to "we propose ...".
   - PARTIAL: the claim credits a DIFFERENT entity than the abstract does (for example a benchmark, dataset or
     competition credited with what a team's method did or achieved).
2. NUMBERS: every number, unit and comparison, digit by digit.
3. MODALITY: only a claim STRONGER than the abstract is a problem ("can be used to estimate" -> "estimates",
   "may" -> "does", "suggests" -> "proves"). A claim that is weaker or more hedged than the abstract
   ("aims to provide" for "provides") is FINE.
4. SCOPE: words like all, always, consistently, any, only, most must be stated in the abstract.
5. ADDED CONTENT: any cause, baseline, dataset, study or number not in the title/abstract makes the claim at best "partial".

PRIORITY RULE ON NUMBERS AND EXACT VALUES:
- If the claim contains a number, percentage, metric value, configuration name or exact value,
  AND that number differs from the abstract (even by a single digit, even a digit swap such as
  0.870 vs 0.970 or 1.736 vs 2.736), then the verdict MUST be "unsupported".
- Never label "partial" a claim whose only problem is a wrong number: "partial" is reserved for cases
  where part of the content is stated and another part is missing, exaggerated, merged, attributed to
  the wrong object, or where the modality is weakened.
- A wrong number is a contradiction with the abstract, not an approximation: it is "unsupported",
  exactly like a reversed result.

EXAMPLES:
- Claim "Faithfulness of 0.870" vs abstract "0.970" -> unsupported (different number).
- Claim "73.7% pass rate" vs abstract "72.7%" -> unsupported (different number).
- Claim "latency of 2.736 ns" vs abstract "1.736 ns" -> unsupported (different number).
- Claim "Coverage Retrieval of 0.942" vs abstract "0.942" -> supported (identical number)."""

PROMPTS = {"v1": SYSTEM, "v2": SYSTEM + CHECKLIST, "v3": SYSTEM + CHECKLIST_V3}

LABELS = {"supported": "etayee", "partial": "partielle", "unsupported": "non_etayee"}


def _norm(text: str) -> str:
    """Normalise for matching: lowercase, unify dashes/quotes/spaces, keep digits and dots in numbers."""
    text = (text.lower().replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-")
            .replace("\u2014", "-"))
    # keep letters, digits, %, and '.' so '0.942' stays '0.942'
    return re.sub(r"[^a-z0-9%.]+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [t.strip(".") for t in _norm(text).split() if t.strip(".")]


ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|\u2026)\s*\]|\.\.\.|\u2026)\s*")


def _fragment_found(fragment: str, source_norm: str, source_tokens: list[str]) -> bool:
    frag = _norm(fragment)
    if not frag:
        return False
    if frag in source_norm:                       # exact match after normalisation
        return True
    ft = _tokens(fragment)
    if len(ft) < 3:
        return False
    src_set = set(source_tokens)
    # a number in the quote must exist in the source: a changed digit is never "formatting"
    if any(any(c.isdigit() for c in t) and t not in src_set for t in ft):
        return False
    sm = SequenceMatcher(None, source_tokens, ft, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(ft) >= 0.85


def evidence_in_abstract(evidence, source: str) -> bool:
    """True if every quoted fragment really appears in the source (title + abstract).

    Anti-hallucination check. Tolerates: several fragments separated by '...', small
    formatting differences, a missing word or two. Never tolerates a different number."""
    if isinstance(evidence, (list, tuple)):
        evidence = " ... ".join(str(x) for x in evidence)
    fragments = [f.strip() for f in ELLIPSIS.split(str(evidence or "")) if f.strip()]
    if not fragments:
        return False
    src_norm, src_tokens = _norm(source), _tokens(source)
    return all(_fragment_found(f, src_norm, src_tokens) for f in fragments)


def verify_claim(claim: Claim, paper: Paper, check_evidence: bool = True,
                 model: str | None = None, prompt: str = "v1") -> Verdict:
    user = f"CLAIM: {claim.text}\n\nTITLE: {paper.title}\n\nABSTRACT:\n{paper.abstract}"
    try:
        out = llm.chat_json(PROMPTS[prompt], user, temperature=0.0, model=model or VERIFIER_MODEL)
    except RuntimeError as e:
        return Verdict(claim_id=claim.id, label="erreur", reason=str(e)[:200], evidence_in_source=False)

    raw = str(out.get("verdict", "")).strip().lower()
    label = LABELS.get(raw, "erreur")
    raw_ev = out.get("evidence", "")
    evidence = " ... ".join(map(str, raw_ev)) if isinstance(raw_ev, list) else str(raw_ev or "").strip()
    found = evidence_in_abstract(evidence, f"{paper.title}. {paper.abstract}")
    reason = str(out.get("reason", "")).strip()

    # a "supported" verdict must be backed by a quote that really exists in the abstract
    if check_evidence and label == "etayee" and not found:
        label = "partielle"
        reason = "[downgraded: evidence not found in title/abstract] " + reason
    return Verdict(claim_id=claim.id, label=label, evidence=evidence, reason=reason,
                   evidence_in_source=found)


def verify_all(claims: list[Claim], papers: list[Paper], progress: bool = False,
               max_errors: int = 5, done: dict | None = None, on_verdict=None, **kw) -> list[Verdict]:
    """done = {claim_id: Verdict} déjà obtenus (reprise) ; on_verdict(v) appelé pour chaque nouveau verdict valide."""
    by_id = {p.id: p for p in papers}
    done = done or {}
    out, streak = [], 0
    for i, c in enumerate(claims, 1):
        if c.id in done:
            out.append(done[c.id])
            if progress:
                print(f"[{i}/{len(claims)}] {c.id} -> {done[c.id].label} (déjà fait)", file=sys.stderr, flush=True)
            continue
        v = verify_claim(c, by_id[c.paper_id], **kw)
        out.append(v)
        if progress:                       # visible on stderr so you can see it is working
            print(f"[{i}/{len(claims)}] {c.id} -> {v.label}", file=sys.stderr, flush=True)
        if v.label == "erreur":
            streak += 1
            if progress:
                print(f"      {v.reason[:300]}", file=sys.stderr, flush=True)
            if streak >= max_errors:
                raise RuntimeError(f"{max_errors} erreurs d'affilée : arrêt, aucun fichier écrit. "
                                   f"Dernière erreur : {v.reason[:300]}")
        else:
            streak = 0
            if on_verdict:
                on_verdict(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=CLAIMS_PATH)
    ap.add_argument("--out", default=VERDICTS_PATH)
    ap.add_argument("--model", default=None, help="remplace config.VERIFIER_MODEL")
    ap.add_argument("--no-evidence-check", dest="check", action="store_false")
    ap.add_argument("--wait-quota", dest="wait_quota", action="store_true",
                    help="si le quota journalier Groq est épuisé, attendre et reprendre automatiquement (peut durer des heures)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1",
                    help="v1 = consigne de base ; v2 = + checklist sujet / chiffres / modalité / portée ; v3 = v2 précisée (sujet = contribution des auteurs, seule une modalité plus forte est une erreur, chiffre faux = unsupported)")
    a = ap.parse_args()

    with open(a.claims, encoding="utf-8") as f:
        data = json.load(f)
    papers = [Paper(**p) for p in data["papers"]]
    claims = [Claim(**c) for c in data["claims"]]

    llm.reset_stats()
    llm.WAIT_FOR_QUOTA = a.wait_quota

    # point de reprise : un Ctrl+C ou une coupure ne fait pas perdre le travail déjà fait
    partial = a.out + ".partial.json"
    cfg_key = {"claims_file": a.claims, "prompt": a.prompt,
               "model": a.model or VERIFIER_MODEL, "check": a.check}
    done: dict[str, Verdict] = {}
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("cfg") == cfg_key:
            done = {v["claim_id"]: Verdict(**v) for v in saved["verdicts"]}
            print(f"Reprise : {len(done)} verdicts déjà faits dans {partial}", file=sys.stderr)
    kept = list(done.values())

    def checkpoint(v: Verdict) -> None:
        kept.append(v)
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg_key, "verdicts": [x.model_dump() for x in kept]},
                      f, ensure_ascii=False)

    try:
        verdicts = verify_all(claims, papers, progress=True, done=done, on_verdict=checkpoint,
                              check_evidence=a.check, model=a.model, prompt=a.prompt)
    except RuntimeError as e:
        sys.exit(f"\nARRET : {e}\n(le travail déjà fait est gardé : relancez la même commande pour reprendre)")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"verifier_model": a.model or VERIFIER_MODEL, "claims_file": a.claims,
                   "evidence_check": a.check, "prompt_version": a.prompt, "verdicts": [v.model_dump() for v in verdicts],
                   "stats": dict(llm.STATS)}, f, ensure_ascii=False, indent=2)

    if os.path.exists(partial):
        os.remove(partial)

    counts = {k: sum(v.label == k for v in verdicts) for k in ("etayee", "partielle", "non_etayee", "erreur")}
    print(f"{len(verdicts)} verdicts {counts} "
          f"| {llm.STATS['prompt_tokens']}+{llm.STATS['completion_tokens']} tokens "
          f"| {llm.STATS['seconds']:.1f}s\n")
    text = {c.id: c.text for c in claims}
    for v in verdicts:
        if v.label != "etayee":
            print(f"{v.claim_id} -> {v.label}\n   claim : {text[v.claim_id]}\n   raison: {v.reason}")


if __name__ == "__main__":
    main()