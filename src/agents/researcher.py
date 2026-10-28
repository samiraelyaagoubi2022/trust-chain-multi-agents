"""J1 v6 : recherche d'articles par pipeline hybride + passe experte + option --source.

Étapes : sujet -> requêtes (LLM) -> arXiv + OpenAlex (ou une seule source) en parallèle
-> filtres -> classement hybride (BM25 + embeddings, fusion RRF) -> juge LLM (notes 0 à 3)
-> expansion par citations (OpenAlex) -> passe experte (requêtes manuelles) -> sélection finale.

Nouveautés v6 :
- Option --source (both, arxiv, openalex) pour lancer une seule source à la fois.
  Utile pour la méthode CMR (Capture-Mark-Recapture) qui a besoin de deux recherches
  indépendantes pour estimer le rappel.

Nouveautés v5 :
- Passe experte : requêtes manuelles ciblées pour rattraper les articles canoniques.

Usage :
    python src\\agents\\researcher.py "retrieval augmented generation evaluation"
Options d'ablation :
    --no-embed  --no-judge  --no-snowball  --no-expert  --tag nom
    --source both|arxiv|openalex
"""
import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")

MIN_YEAR = 2022
ARXIV_CATEGORIES = ["cs.CL", "cs.IR", "cs.AI", "cs.LG"]
MIN_ABSTRACT_CHARS = 300
PER_QUERY = 10
JUDGE_TOP = 24
MIN_SCORE = 2
MAX_KEEP = 12
TARGET_KEEP = 8
MAX_ROUNDS = 2
DEFAULT_JUDGE_MODE = "score"
SNOWBALL_SEEDS = 3
SNOWBALL_PER_SEED = 6

EXPERT_QUERIES = {
    "retrieval augmented generation evaluation": [
        "RAGAS automated evaluation",
        "RAGBench benchmark",
        "ARES evaluation framework",
        "RAGChecker framework",
        "MIRAGE benchmark",
        "CRAG benchmark",
        "RAG evaluation survey",
        "RAG faithfulness metric",
    ],
}

OA_TYPES = {"article", "preprint", "review"}
OA_TYPE_FILTER = "type:" + "|".join(sorted(OA_TYPES))
BAD_URL_PARTS = ("zenodo", "10.5281", "figshare", "10.6084", "osf.io", "10.17605")
NON_PAPER_RE = re.compile(
    r"(supporting materials?|supplementary materials?|replication package|"
    r"this repository|this dataset|evaluation harness|slides? (for|from))",
    re.IGNORECASE,
)

STOP = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with", "by", "from", "using"}


@dataclass
class Paper:
    source: str
    title: str
    abstract: str
    url: str
    year: int | None = None
    query: str = ""
    oa_id: str = ""
    score: float | None = None
    relevance: int | None = None
    judge_why: str = ""


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def looks_like_paper(p: Paper) -> bool:
    url = (p.url or "").lower()
    if any(part in url for part in BAD_URL_PARTS):
        return False
    if NON_PAPER_RE.search(p.abstract[:200]):
        return False
    return True


def http_get(url: str, params: dict | None = None, tries: int = 3) -> requests.Response:
    params = dict(params or {})
    if "openalex.org" in url and OPENALEX_EMAIL:
        params.setdefault("mailto", OPENALEX_EMAIL)
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep(2 * (k + 1))
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise requests.RequestException("unreachable")


# ==========================================================================
# LLM : sujet, requêtes, juge
# ==========================================================================
def llm_json(prompt: str, temperature: float = 0.2) -> dict:
    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(resp.choices[0].message.content)


def to_english_topic(topic: str) -> str:
    prompt = (
        "Rewrite this research topic as a short English noun phrase (max 12 words), "
        "keeping every technical term.\n"
        f'Topic: "{topic}"\n'
        'Answer only with JSON: {"topic_en": "..."}'
    )
    try:
        return llm_json(prompt)["topic_en"].strip() or topic
    except Exception as e:
        warn(f"Reformulation du sujet impossible ({e}) : sujet gardé tel quel.")
        return topic


def reformulate(topic_en: str, source: str, previous: list[str] | None = None, n: int = 3) -> list[str]:
    hint = ""
    if previous:
        hint = (
            "Previous queries returned too few relevant papers: " + "; ".join(previous)
            + ". Propose DIFFERENT queries (other synonyms, related techniques, "
            "benchmark or metric names).\n"
        )
    prompt = (
        f"Research topic: {topic_en}\n{hint}"
        f"Write {n} search queries for {source}. Rules:\n"
        "- each query is 2 to 5 words, in English, made of precise technical terms;\n"
        "- name the real technologies, methods, metrics or benchmarks of the field "
        "(never generic words such as 'performance', 'study' or 'approach');\n"
        "- no quotes, no boolean operators.\n"
        'Answer only with JSON: {"queries": ["...", "..."]}'
    )
    try:
        queries = llm_json(prompt)["queries"]
        queries = [q.strip().strip('"') for q in queries if isinstance(q, str) and q.strip()]
        return queries[:n] or [topic_en]
    except Exception as e:
        warn(f"[{source}] génération de requêtes impossible ({e}) : sujet utilisé.")
        return [topic_en]


JUDGE_BATCH = 10


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _judge_chunk(topic_en: str, papers: list[Paper], criteria: str = "") -> list[int]:
    if not papers:
        return []
    listing = "\n".join(f"[{i}] {p.title} :: {p.abstract[:600]}" for i, p in enumerate(papers))
    rule = (
        f"Relevance criteria: {criteria}\n" if criteria
        else "A paper is relevant if its main contribution is about the topic itself, "
             "not merely an application or a mention.\n"
    )
    prompt = (
        f"Topic: {topic_en}\n{rule}"
        "Score each item from 0 to 3 for how directly it addresses the topic.\n"
        "3 = core contribution on the topic, 2 = clearly relevant, "
        "1 = only mentions it or tangential, 0 = off-topic. Be strict and consistent.\n"
        "Also give 0 to any item that is not a real research paper with a method or "
        "findings (dataset or repository record, thesis supplement, slides).\n\n"
        f"{listing}\n\n"
        'Answer only with JSON: {"scores": [{"i": 0, "score": 3}]}'
    )
    scores = [0] * len(papers)
    try:
        for item in llm_json(prompt, temperature=0.0)["scores"]:
            i, s = item.get("i"), item.get("score")
            if isinstance(i, int) and 0 <= i < len(papers) and isinstance(s, (int, float)):
                scores[i] = int(s)
    except Exception as e:
        warn(f"Juge indisponible ({e}) : articles conservés sans filtre.")
        return [MIN_SCORE] * len(papers)
    return scores


def judge(topic_en: str, papers: list[Paper], criteria: str = "") -> list[int]:
    scores: list[int] = []
    for chunk in _chunks(papers, JUDGE_BATCH):
        scores += _judge_chunk(topic_en, chunk, criteria)
    return scores


def _b(x) -> bool:
    return x is True or str(x).strip().lower() == "true"


def score_from_flags(f: dict) -> int:
    if not _b(f.get("real_paper", True)) or _b(f.get("survey_or_doc")):
        return 0
    score = 3 if _b(f.get("core_on_topic")) else 2 if _b(f.get("partly_on_topic")) else 1
    if _b(f.get("single_domain")) or _b(f.get("new_method_only")):
        score = min(score, 1)
    return score


def _structured_call(topic_en: str, papers: list[Paper], criteria: str) -> dict[int, dict]:
    listing = "\n".join(f"[{i}] {p.title} :: {p.abstract[:600]}" for i, p in enumerate(papers))
    rule = f"Relevance criteria: {criteria}\n" if criteria else ""
    prompt = (
        f"Topic: {topic_en}\n{rule}"
        f"There are {len(papers)} papers, numbered 0 to {len(papers) - 1}. Answer for EVERY one of them.\n"
        "For each paper answer these questions with true or false:\n"
        "- real_paper: a research paper with a method or findings (false for dataset/repository records, slides)\n"
        "- core_on_topic: its MAIN contribution is about the topic itself\n"
        "- partly_on_topic: the topic is a significant part of it, though not the main contribution\n"
        "- single_domain: tied to ONE application domain or language group (medicine, law, soil science...). "
        "Answer true ONLY if the criteria above exclude domain-specific work.\n"
        "- new_method_only: mainly proposes a new system/method instead of studying or evaluating the topic\n"
        "- survey_or_doc: a survey, tutorial or product documentation\n"
        "Add \"why\": one short sentence. Judge from the abstract only.\n\n"
        f"{listing}\n\n"
        'Answer only with JSON: {"papers": [{"i": 0, "real_paper": true, "core_on_topic": true, '
        '"partly_on_topic": false, "single_domain": false, "new_method_only": false, '
        '"survey_or_doc": false, "why": "..."}]}'
    )
    out: dict[int, dict] = {}
    for item in llm_json(prompt, temperature=0.0)["papers"]:
        i = item.get("i")
        if isinstance(i, int) and 0 <= i < len(papers):
            out[i] = item
    return out


def _judge_structured_chunk(topic_en: str, chunk: list[Paper], criteria: str) -> list[int]:
    flags: dict[int, dict] = {}
    failed = False
    for _ in range(2):
        pending = [i for i in range(len(chunk)) if i not in flags]
        if not pending:
            break
        try:
            got = _structured_call(topic_en, [chunk[i] for i in pending], criteria)
        except Exception as e:
            warn(f"Juge structuré indisponible ({e}).")
            failed = True
            break
        for j, item in got.items():
            flags[pending[j]] = item
    if failed and not flags:
        for p in chunk:
            p.judge_why = "judge unavailable"
        return [MIN_SCORE] * len(chunk)

    scores, missing = [], 0
    for i, p in enumerate(chunk):
        f = flags.get(i)
        if f is None:
            scores.append(0)
            p.judge_why = "no answer from judge"
            missing += 1
            continue
        scores.append(score_from_flags(f))
        p.judge_why = str(f.get("why", ""))[:200]
    if missing:
        warn(f"[juge] {missing} article(s) sans réponse après 2 essais : résultat à vérifier.")
    return scores


def judge_structured(topic_en: str, papers: list[Paper], criteria: str = "") -> list[int]:
    scores: list[int] = []
    for chunk in _chunks(papers, JUDGE_BATCH):
        scores += _judge_structured_chunk(topic_en, chunk, criteria)
    return scores


# ==========================================================================
# Sources : arXiv, OpenAlex, citations OpenAlex
# ==========================================================================
def search_arxiv(query: str, n: int) -> list[Paper]:
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1 and t.lower() not in STOP]

    def fetch(used: list[str]) -> list[Paper]:
        clause = " AND ".join(f"abs:{t}" for t in used)
        if ARXIV_CATEGORIES:
            clause += " AND (" + " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES) + ")"
        r = http_get(
            "https://export.arxiv.org/api/query",
            {"search_query": clause, "max_results": n, "sortBy": "relevance"},
        )
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            published = e.findtext("a:published", default="", namespaces=ns)
            out.append(Paper(
                source="arxiv",
                title=" ".join(e.findtext("a:title", default="", namespaces=ns).split()),
                abstract=" ".join(e.findtext("a:summary", default="", namespaces=ns).split()),
                url=e.findtext("a:id", default="", namespaces=ns),
                year=int(published[:4]) if published[:4].isdigit() else None,
            ))
        return out

    papers = fetch(terms)
    if not papers and len(terms) > 3:
        papers = fetch(terms[:3])
    return papers


def _rebuild_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    words = {pos: w for w, positions in inverted.items() for pos in positions}
    return " ".join(words[i] for i in sorted(words))


def _openalex_to_paper(w: dict) -> Paper | None:
    wtype = w.get("type")
    if wtype and wtype not in OA_TYPES:
        return None
    abstract = _rebuild_abstract(w.get("abstract_inverted_index"))
    if not abstract:
        return None
    p = Paper(
        source="openalex",
        title=w.get("title") or "",
        abstract=abstract,
        url=w.get("doi") or w.get("id") or "",
        year=w.get("publication_year"),
        oa_id=w.get("id") or "",
    )
    return p if looks_like_paper(p) else None


def search_openalex(query: str, n: int) -> list[Paper]:
    r = http_get(
        "https://api.openalex.org/works",
        {
            "search": query,
            "per-page": n,
            "filter": f"has_abstract:true,from_publication_date:{MIN_YEAR}-01-01,{OA_TYPE_FILTER}",
        },
    )
    return [p for p in map(_openalex_to_paper, r.json().get("results", [])) if p]


def _fetch_works(filter_str: str, per_page: int, tag: str) -> list[Paper]:
    r = http_get(
        "https://api.openalex.org/works",
        {"filter": filter_str, "sort": "cited_by_count:desc", "per-page": per_page},
    )
    out = []
    for w in r.json().get("results", []):
        p = _openalex_to_paper(w)
        if p:
            p.query = tag
            out.append(p)
    return out


def snowball(seeds: list[Paper]) -> list[Paper]:
    out: list[Paper] = []
    base = f"has_abstract:true,from_publication_date:{MIN_YEAR}-01-01,{OA_TYPE_FILTER}"
    for s in seeds:
        wid = s.oa_id.rsplit("/", 1)[-1]
        try:
            out += _fetch_works(f"cites:{wid},{base}", SNOWBALL_PER_SEED, f"cite {wid}")
        except requests.RequestException as e:
            warn(f"[citations avant] {wid} échoué : {e}")
        try:
            r = http_get(f"https://api.openalex.org/works/{wid}", {"select": "referenced_works"})
            refs = [x.rsplit("/", 1)[-1] for x in r.json().get("referenced_works", [])][:50]
            if refs:
                out += _fetch_works(f"openalex:{'|'.join(refs)},{base}", SNOWBALL_PER_SEED, f"ref {wid}")
        except requests.RequestException as e:
            warn(f"[citations arrière] {wid} échoué : {e}")
        time.sleep(0.5)
    return out


def resolve_oa_id(p: Paper) -> str:
    try:
        r = http_get(
            "https://api.openalex.org/works",
            {"search": p.title, "per-page": 3, "select": "id,title"},
        )
    except requests.RequestException as e:
        warn(f"[citations] identifiant introuvable pour '{p.title[:40]}' : {e}")
        return ""
    for w in r.json().get("results", []):
        if norm(w.get("title") or "") == norm(p.title):
            return w.get("id") or ""
    return ""


SEARCHERS = {"arxiv": search_arxiv, "openalex": search_openalex}


def retrieve(source: str, topic_en: str, tried: list[str]) -> list[Paper]:
    queries = reformulate(topic_en, source, tried or None)
    if topic_en not in tried:
        queries = [topic_en] + queries
    tried.extend(queries)
    found: list[Paper] = []
    for q in queries:
        try:
            results = SEARCHERS[source](q, PER_QUERY)
        except requests.RequestException as e:
            warn(f"[{source}] requête '{q}' échouée : {e}")
            continue
        for p in results:
            p.query = q
        found.extend(results)
        time.sleep(1.0)
    return found


# ==========================================================================
# Classement hybride : BM25 + embeddings, fusion RRF
# ==========================================================================
def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    avgdl = (sum(len(t) for t in toks) / n) or 1.0
    df: Counter = Counter()
    for t in toks:
        df.update(set(t))
    q_terms = set(tokenize(query))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q_terms:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name=EMBED_MODEL)
        except Exception as e:
            warn(f"Embeddings indisponibles ({e}) : classement BM25 seul.")
            _embedder = False
    return _embedder


def dense_scores(query: str, docs: list[str]) -> list[float] | None:
    emb = get_embedder()
    if not emb:
        return None
    import numpy as np
    vecs = np.array(list(emb.embed([query] + docs)), dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs[1:] @ vecs[0]).tolist()


def rrf(rankings: list[list[int]], n_items: int, k: int = 60) -> list[float]:
    fused = [0.0] * n_items
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


def hybrid_rank(topic_en: str, papers: list[Paper], use_embed: bool = True) -> list[Paper]:
    if not papers:
        return []
    docs = [f"{p.title}. {p.abstract[:1000]}" for p in papers]
    idx = range(len(papers))
    bm = bm25_scores(topic_en, docs)
    rankings = [sorted(idx, key=lambda i: bm[i], reverse=True)]
    if use_embed:
        dense = dense_scores(topic_en, docs)
        if dense is not None:
            rankings.append(sorted(idx, key=lambda i: dense[i], reverse=True))
    fused = rrf(rankings, len(papers))
    for p, s in zip(papers, fused):
        p.score = round(s, 5)
    return sorted(papers, key=lambda p: p.score or 0, reverse=True)


# ==========================================================================
# Pipeline
# ==========================================================================
def add_to_pool(pool: dict[str, Paper], papers: list[Paper]) -> int:
    added = 0
    for p in papers:
        key = norm(p.title)
        if not key:
            continue
        if key in pool:
            if p.oa_id and not pool[key].oa_id:
                pool[key].oa_id = p.oa_id
            continue
        if (len(p.abstract) >= MIN_ABSTRACT_CHARS
                and (p.year is None or p.year >= MIN_YEAR)
                and looks_like_paper(p)):
            pool[key] = p
            added += 1
    return added


def select(topic_en, pool, judged, cfg):
    criteria = getattr(cfg, "criteria", "") or ""
    rank_query = f"{topic_en}. {criteria}" if criteria else topic_en
    ranked = hybrid_rank(rank_query, list(pool.values()), cfg.embed)
    top = ranked[:getattr(cfg, "judge_top", JUDGE_TOP)]
    if not cfg.judge:
        return top[:MAX_KEEP], []
    todo = [p for p in top if norm(p.title) not in judged]
    mode = getattr(cfg, "judge_mode", DEFAULT_JUDGE_MODE)
    judge_fn = judge_structured if mode == "structured" else judge
    for p, s in zip(todo, judge_fn(topic_en, todo, criteria)):
        judged[norm(p.title)] = s
    for p in top:
        p.relevance = judged.get(norm(p.title), 0)
    kept = [p for p in top if p.relevance >= MIN_SCORE]
    rejected = [p for p in top if p.relevance < MIN_SCORE]
    kept.sort(key=lambda p: (p.relevance, p.score or 0), reverse=True)
    return kept[:MAX_KEEP], rejected


def expert_pass(pool: dict[str, Paper], topic_en: str, judged: dict[str, int], cfg) -> int:
    expert = EXPERT_QUERIES.get(topic_en.lower(), [])
    if not expert:
        return 0
    added = 0
    # [WARNING] v6 : la passe experte respecte --source
    sources = list(SEARCHERS.keys()) if getattr(cfg, "source", "both") == "both" else [cfg.source]
    for source in sources:
        for q in expert:
            try:
                results = SEARCHERS[source](q, PER_QUERY)
            except requests.RequestException as e:
                warn(f"[expert] requête '{q}' échouée sur {source} : {e}")
                continue
            for p in results:
                p.query = f"expert: {q}"
            added += add_to_pool(pool, results)
            time.sleep(0.5)
    return added


def run(topic: str, cfg):
    t0 = time.time()
    pool: dict[str, Paper] = {}
    # [WARNING] v6 : tried est limité aux sources choisies
    sources = list(SEARCHERS.keys()) if getattr(cfg, "source", "both") == "both" else [cfg.source]
    tried = {s: [] for s in sources}
    judged: dict[str, int] = {}
    kept, rejected, stages = [], [], []

    if cfg.pool:
        with open(cfg.pool, encoding="utf-8") as f:
            saved = json.load(f)
        topic_en = saved["topic_en"]
        dropped = 0
        for d in saved["papers"]:
            d["score"], d["relevance"], d["judge_why"] = None, None, ""
            p = Paper(**d)
            if not looks_like_paper(p):
                dropped += 1
                continue
            pool[norm(p.title)] = p
        print(f"Sujet de recherche : {topic_en} (pool chargé : {len(pool)} candidats, "
              f"{dropped} écartés comme non-articles)")
        kept, rejected = select(topic_en, pool, judged, cfg)
        stages.append({"depuis_pool": len(pool), "ecartes_non_articles": dropped, "retenus": len(kept)})
        rounds = range(0)
    else:
        topic_en = to_english_topic(topic)
        print(f"Sujet de recherche : {topic_en}")
        print(f"Sources : {', '.join(sources)}")
        rounds = range(1, MAX_ROUNDS + 1)

    for rnd in rounds:
        with ThreadPoolExecutor(max_workers=len(sources)) as ex:
            futures = [ex.submit(retrieve, s, topic_en, tried[s]) for s in sources]
            fetched = [f.result() for f in futures]
        added = add_to_pool(pool, [p for res in fetched for p in res])
        kept, rejected = select(topic_en, pool, judged, cfg)
        stage = {"tour": rnd, "nouveaux_candidats": added, "pool": len(pool), "retenus": len(kept)}

        if cfg.snowball and rnd == 1 and kept:
            for p in kept[:SNOWBALL_SEEDS]:
                if not p.oa_id:
                    p.oa_id = resolve_oa_id(p)
            seeds = [p for p in kept[:SNOWBALL_SEEDS] if p.oa_id]
            added_snow = add_to_pool(pool, snowball(seeds))
            if added_snow:
                kept, rejected = select(topic_en, pool, judged, cfg)
            stage.update(par_citations=added_snow, pool=len(pool), retenus=len(kept))

        stages.append(stage)
        print(f"Tour {rnd} : {stage}")
        if len(kept) >= TARGET_KEEP:
            break

    if not cfg.pool and rounds and getattr(cfg, "expert", True):
        added_expert = expert_pass(pool, topic_en, judged, cfg)
        if added_expert:
            kept, rejected = select(topic_en, pool, judged, cfg)
            stages.append({"passe_experte": added_expert, "pool": len(pool), "retenus": len(kept)})
            print(f"Passe experte : +{added_expert} candidats, pool={len(pool)}, retenus={len(kept)}")

    if not cfg.pool:
        os.makedirs("data", exist_ok=True)
        suffix = f"_{cfg.tag}" if cfg.tag else ""
        with open(f"data/pool{suffix}.json", "w", encoding="utf-8") as f:
            json.dump({"topic_en": topic_en, "papers": [asdict(p) for p in pool.values()]},
                      f, ensure_ascii=False, indent=2)

    log = {
        "sujet": topic, "sujet_en": topic_en, "requetes": tried, "etapes": stages,
        "config": {"embed": cfg.embed, "judge": cfg.judge, "snowball": cfg.snowball,
                   "expert": getattr(cfg, "expert", True), "source": getattr(cfg, "source", "both"),
                   "criteria": getattr(cfg, "criteria", ""), "judge_mode": getattr(cfg, "judge_mode", DEFAULT_JUDGE_MODE),
                   "judge_top": getattr(cfg, "judge_top", JUDGE_TOP),
                   "llm_model": LLM_MODEL,
                   "oa_types": sorted(OA_TYPES)},
        "duree_s": round(time.time() - t0, 1),
    }
    return kept, rejected, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", nargs="*")
    ap.add_argument("--no-embed", dest="embed", action="store_false")
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--no-snowball", dest="snowball", action="store_false")
    ap.add_argument("--no-expert", dest="expert", action="store_false",
                    help="désactive la passe experte (pour ablation)")
    ap.add_argument("--tag", default="", help="suffixe des fichiers de sortie")
    ap.add_argument("--source", default="both", choices=["both", "arxiv", "openalex"],
                    help="source unique (both = arXiv + OpenAlex). Utile pour CMR.")
    ap.add_argument("--judge-mode", dest="judge_mode", choices=["score", "structured"],
                    default=DEFAULT_JUDGE_MODE,
                    help="score = note globale ; structured = questions oui/non + règle fixe")
    ap.add_argument("--judge-top", dest="judge_top", type=int, default=JUDGE_TOP,
                    help="nombre de candidats (les mieux classés) transmis au juge")
    ap.add_argument("--criteria", default="",
                    help="définition précise de la pertinence, donnée au juge et au classement")
    ap.add_argument("--pool", default=None,
                    help="reclasse un pool de candidats déjà sauvegardé (data/pool_xxx.json), sans nouvelle recherche")
    cfg = ap.parse_args()

    topic = " ".join(cfg.topic) or "retrieval augmented generation evaluation"
    kept, rejected, log = run(topic, cfg)

    print(f"\n{len(kept)} articles retenus, {len(rejected)} rejetés, {log['duree_s']} s\n")
    if not kept and cfg.judge:
        print("Aucun article jugé pertinent : vérifiez que le sujet est précis et bien écrit.")
    for p in kept:
        why = f"  [{p.judge_why}]" if p.judge_why else ""
        print(f"- [{p.source}] note {p.relevance} | {p.title} ({p.year}){why}")
    print("\nExemples de rejetés :")
    for p in rejected[:8]:
        why = f"  ({p.judge_why})" if p.judge_why else ""
        print(f"  x [{p.source}] note {p.relevance} | {p.title}{why}")

    suffix = f"_{cfg.tag}" if cfg.tag else ""
    os.makedirs("data", exist_ok=True)
    for name, content in (("papers", [asdict(p) for p in kept]),
                          ("rejected", [asdict(p) for p in rejected]),
                          ("search_log", log)):
        with open(f"data/{name}{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)