"""Assemble le jeu d'évaluation du pipeline : vraies affirmations + erreurs, avec leur label connu.

    python scripts/make_eval_set.py
    -> eval/claims_evalset.json   (articles + affirmations, à donner à `python -m src.pipeline --claims ...`)
    -> eval/gold_evalset.json     (labels : vos annotations + labels des erreurs synthétiques)

Sources : samefamily (30, dont vos 3 erreurs naturelles), synthetic (9), synthetic2 (9).
Les identifiants sont préfixés (F = samefamily, X = synthetic, Y = synthetic2) pour éviter les collisions.
"""
import json
import re

SOURCES = [("samefamily", "eval/claims_samefamily.json", "eval/gold_samefamily.json", "F"),
           ("synthetic", "eval/claims_synthetic.json", "eval/gold_synthetic.json", "X"),
           ("synthetic2", "eval/claims_synthetic2.json", "eval/gold_synthetic2.json", "Y")]
LABELS = ("etayee", "partielle", "non_etayee")


def build(sources):
    papers: dict[str, dict] = {}
    claims, meta, annotations = [], {}, []
    topic = ""
    for name, claims_path, gold_path, prefix in sources:
        with open(claims_path, encoding="utf-8") as f:
            data = json.load(f)
        with open(gold_path, encoding="utf-8") as f:
            gold = {a["claim_id"]: a for a in json.load(f)["annotations"]}
        originals = {m["id"]: m["original"] for m in data.get("synthetic", [])}
        topic = topic or data.get("topic", "")
        for p in data["papers"]:
            if p["id"] in papers and papers[p["id"]]["title"] != p["title"]:
                raise ValueError(f"l'identifiant {p['id']} désigne deux articles différents")
            papers[p["id"]] = p
        for c in data["claims"]:
            label = (gold.get(c["id"], {}).get("human_label") or "").strip()
            if label not in LABELS:
                print(f"ignorée (pas de label) : {name}:{c['id']}")
                continue
            new_id = prefix + re.sub(r"\D", "", c["id"])
            claims.append({"id": new_id, "text": c["text"], "paper_id": c["paper_id"]})
            meta[new_id] = {"gold": label, "source": name, "orig_id": c["id"],
                            "original_text": originals.get(c["id"])}
            annotations.append({"claim_id": new_id, "claim": c["text"], "human_label": label,
                                "note": f"{name}:{c['id']}"})
    doc = {"topic": topic, "writer_model": "mixed", "papers": list(papers.values()),
           "claims": claims, "meta": meta}
    return doc, {"claims_file": "eval/claims_evalset.json", "annotations": annotations}


def main():
    doc, gold = build(SOURCES)
    with open("eval/claims_evalset.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    with open("eval/gold_evalset.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    n_err = sum(1 for m in doc["meta"].values() if m["gold"] != "etayee")
    print(f"{len(doc['claims'])} affirmations : {len(doc['claims']) - n_err} étayées, {n_err} avec erreur "
          f"-> eval/claims_evalset.json et eval/gold_evalset.json")


if __name__ == "__main__":
    main()