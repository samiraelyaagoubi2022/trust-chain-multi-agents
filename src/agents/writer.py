"""J2 - Writer: reads abstracts and produces numbered claims.

Run from the project root:
    python -m src.agents.writer --topic "evaluation of RAG systems"
"""
import argparse
import json
import re
import sys

from src import llm
from src.config import CLAIMS_PATH, MAX_PAPERS, MIN_RELEVANCE, MODEL, PAPERS_PATH
from src.schemas import Claim, Paper

SYSTEM = """You are a rigorous scientific writer. You write a synthesis note in English
based on a list of papers (title + abstract), and ONLY on them.

Strict rules:
- Each claim expresses ONE single idea, in one sentence.
- Each claim cites ONE single paper (its identifier, e.g. "P3").
- Only write a number, a method name or a result if it appears in the abstract of the cited paper.
- Add no outside knowledge and make no generalization across papers.
- If an abstract says nothing precise, do not invent: write fewer claims.

Answer only with this JSON:
{"claims": [{"text": "...", "paper_id": "P1"}, ...]}"""


PLAIN_SYSTEM = """You are a helpful assistant that summarises scientific papers for a busy reader.
For each paper, write informative claims about its contribution, its results and why it matters. Each claim is
one sentence and cites one paper by its identifier (e.g. "P3"). Be specific and concrete.

Answer only with this JSON:
{"claims": [{"text": "...", "paper_id": "P1"}, ...]}"""

STYLES = {"strict": SYSTEM, "plain": PLAIN_SYSTEM}   # plain = baseline writer WITHOUT anti-hallucination rules

# Some models drift into another script mid-sentence (e.g. "multi-step检索 pipeline").
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\u0590-\u06FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")


def load_papers(path: str = PAPERS_PATH) -> list[Paper]:
    """Load papers.json, filter weak entries, assign ids P1, P2..."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw = [p for p in raw
           if (p.get("relevance") or MIN_RELEVANCE) >= MIN_RELEVANCE
           and len(p.get("abstract", "")) >= 300
           and "zenodo" not in p.get("url", "")]
    raw = raw[:MAX_PAPERS]
    return [Paper(id=f"P{i}", title=p["title"], abstract=p["abstract"],
                  url=p.get("url", ""), year=p.get("year"), source=p.get("source", ""))
            for i, p in enumerate(raw, 1)]


def write_note(topic: str, papers: list[Paper], per_paper: int = 2,
               model: str | None = None, style: str = "strict") -> list[Claim]:
    """One LLM call -> list of validated claims."""
    corpus = "\n\n".join(f"[{p.id}] {p.title} ({p.year})\n{p.abstract}" for p in papers)
    user = (f"Topic: {topic}\n\nPapers:\n{corpus}\n\n"
            f"Write about {per_paper} claims per paper, the most informative ones.")
    data = llm.chat_json(STYLES[style], user, temperature=0.3, model=model)

    valid_ids = {p.id for p in papers}
    claims, n = [], 1
    for c in data.get("claims", []):
        # guard: drop claims citing an unknown paper, empty, or containing foreign-script characters
        text = str(c.get("text", "")).strip()
        if c.get("paper_id") not in valid_ids or not text:
            continue
        if FOREIGN_SCRIPT.search(text):
            print(f"[writer] claim dropped (foreign characters): {text[:80]}", file=sys.stderr)
            continue
        claims.append(Claim(id=f"C{n}", text=text, paper_id=c["paper_id"]))
        n += 1
    return claims


REVISE_SYSTEM = """You are a rigorous scientific writer correcting your own claims after a fact-check.
For each claim you receive: the claim, the fact-checker's objection, and the title and abstract of the cited paper.
Rewrite the claim so that EVERY element of it (numbers, names, who does what, strength of the wording) is stated
in the title/abstract. Keep it to one sentence and one idea. Do not change the cited paper.
If the idea cannot be stated faithfully from the abstract, set "text" to null: the claim will be deleted.
Add nothing that is not in the abstract.
Answer only with JSON: {"claims": [{"id": "C5", "text": "..." or null}, ...]}"""

REVIEW_SYSTEM = """You are reviewing claims you wrote earlier. For each claim, re-read the title and abstract of the
cited paper and decide whether EVERY element of the claim (numbers, names, who does what, strength of the wording)
is stated there. If yes, return the claim text unchanged. If not, rewrite it so that it is fully supported by the
abstract, or set "text" to null if it cannot be supported. Add nothing that is not in the abstract.
Answer only with JSON: {"claims": [{"id": "C5", "text": "..." or null}, ...]}"""


def _ask_rewrite(system: str, blocks: list[str], ids: set[str], model: str | None) -> dict[str, str | None]:
    """One LLM call. Returns {claim_id: new text, or None = delete}. Claims not answered are absent."""
    data = llm.chat_json(system, "\n\n".join(blocks), temperature=0.2, model=model)
    out: dict[str, str | None] = {}
    for item in data.get("claims", []):
        cid = item.get("id")
        if cid not in ids or "text" not in item:
            continue
        text = item["text"]
        if text is None or not str(text).strip():
            out[cid] = None
            continue
        text = str(text).strip()
        if FOREIGN_SCRIPT.search(text):          # same guard as for first drafts: keep the claim unchanged
            print(f"[writer] rewrite ignored (foreign characters): {text[:80]}", file=sys.stderr)
            continue
        out[cid] = text
    return out


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def revise(items: list[tuple[Claim, str]], papers: list[Paper], model: str | None = None,
           batch: int = 8) -> dict[str, str | None]:
    """Correct rejected claims. items = [(claim, verifier's reason)]. None = delete the claim."""
    by_id = {p.id: p for p in papers}
    out: dict[str, str | None] = {}
    for chunk in _chunks(items, batch):
        blocks = [f"[{c.id}] CLAIM: {c.text}\nOBJECTION: {reason}\n"
                  f"PAPER {by_id[c.paper_id].id}: {by_id[c.paper_id].title}\n"
                  f"ABSTRACT: {by_id[c.paper_id].abstract}" for c, reason in chunk]
        out.update(_ask_rewrite(REVISE_SYSTEM, blocks, {c.id for c, _ in chunk}, model))
    return out


def self_review(claims: list[Claim], papers: list[Paper], model: str | None = None,
                batch: int = 10) -> dict[str, str | None]:
    """Baseline: the writer re-reads its own claims, with the same context and no outside verdict."""
    by_id = {p.id: p for p in papers}
    out: dict[str, str | None] = {}
    for chunk in _chunks(claims, batch):
        blocks = [f"[{c.id}] CLAIM: {c.text}\nPAPER {by_id[c.paper_id].id}: {by_id[c.paper_id].title}\n"
                  f"ABSTRACT: {by_id[c.paper_id].abstract}" for c in chunk]
        out.update(_ask_rewrite(REVIEW_SYSTEM, blocks, {c.id for c in chunk}, model))
    return out


SYNTHESIZE_SYSTEM = """You are a rigorous scientific writer. 
You are given a research topic and a list of VERIFIED CLAIMS (extracted from papers and fact-checked).
Write a structured, coherent literature review in English based ONLY on these claims.

Strict rules:
- Do NOT add any outside knowledge. Use ONLY the provided facts.
- Organize the facts logically into paragraphs (e.g., Introduction, Methods, Benchmarks).
- Keep the academic tone.
- Cite the paper IDs (e.g., [P1], [P3]) in the text where the facts come from.
"""

def synthesize(topic: str, verified_claims: list[Claim], model: str | None = None) -> str:
    """Agent 3 (Action Synthèse) : Rédige la note finale à partir des faits vérifiés."""
    if not verified_claims:
        return "No verified claims to synthesize."

    facts_text = "\n".join(f"- {c.text} [{c.paper_id}]" for c in verified_claims)
    user = f"Topic: {topic}\n\nVerified Facts:\n{facts_text}\n\nWrite the literature review:"
    
    resp = llm.get_client().chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZE_SYSTEM},
            {"role": "user", "content": user}
        ],
        temperature=0.3
    )
    
    llm.STATS['prompt_tokens'] += resp.usage.prompt_tokens
    llm.STATS['completion_tokens'] += resp.usage.completion_tokens
    
    return resp.choices[0].message.content


if __name__ == "__main__":
    main()