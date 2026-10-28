"""Fusionne plusieurs fichiers de claims en un seul (déduplication par id).

Usage :
    python scripts/merge_claims.py eval/claims_evalset.json eval/claims_more.json --out eval/claims_merged.json

Règles :
- Les articles sont dédupliqués par id (P1, P2...).
- Les claims sont dédupliqués par id (F1, C1...).
- Le topic et le writer_model viennent du premier fichier.
"""
import argparse
import json


def merge(paths: list[str], out: str) -> None:
    papers: dict[str, dict] = {}
    claims: dict[str, dict] = {}
    topic = ""
    writer_model = ""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        topic = topic or data.get("topic", "")
        writer_model = writer_model or data.get("writer_model", "")
        for p in data["papers"]:
            papers[p["id"]] = p
        for c in data["claims"]:
            claims[c["id"]] = c
    merged = {
        "topic": topic,
        "writer_model": writer_model,
        "papers": list(papers.values()),
        "claims": list(claims.values()),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"{len(merged['claims'])} claims, {len(merged['papers'])} papers -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a.inputs, a.out)


if __name__ == "__main__":
    main()