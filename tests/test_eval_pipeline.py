import json

from scripts import evaluate_pipeline as ep
from scripts import make_eval_set as mes


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_build_prefixes_ids_merges_papers_and_keeps_labels(tmp_path):
    paper = {"id": "P1", "title": "T", "abstract": "a" * 300}
    c1 = _write(tmp_path, "c1.json", {"topic": "t", "papers": [paper],
                                       "claims": [{"id": "C1", "text": "true", "paper_id": "P1"},
                                                  {"id": "C2", "text": "unlabelled", "paper_id": "P1"}]})
    g1 = _write(tmp_path, "g1.json", {"annotations": [{"claim_id": "C1", "human_label": "etayee"},
                                                      {"claim_id": "C2", "human_label": ""}]})
    c2 = _write(tmp_path, "c2.json", {"papers": [paper], "claims": [{"id": "S3", "text": "false 0.9", "paper_id": "P1"}],
                                       "synthetic": [{"id": "S3", "base_claim_id": "C1", "kind": "number_changed",
                                                      "original": "true"}]})
    g2 = _write(tmp_path, "g2.json", {"annotations": [{"claim_id": "S3", "human_label": "non_etayee"}]})
    doc, gold = mes.build([("a", c1, g1, "F"), ("b", c2, g2, "X")])
    assert [c["id"] for c in doc["claims"]] == ["F1", "X3"] and len(doc["papers"]) == 1
    assert doc["meta"]["X3"] == {"gold": "non_etayee", "source": "b", "orig_id": "S3", "original_text": "true"}
    assert [a["human_label"] for a in gold["annotations"]] == ["etayee", "non_etayee"]


def _evalset():
    return {"claims": [{"id": "F1", "text": "good", "paper_id": "P1"},
                       {"id": "F2", "text": "good two", "paper_id": "P1"},
                       {"id": "X1", "text": "wrong 0.9", "paper_id": "P1"},
                       {"id": "X2", "text": "wrong 12%", "paper_id": "P1"},
                       {"id": "F3", "text": "natural error", "paper_id": "P1"}],
            "meta": {"F1": {"gold": "etayee"}, "F2": {"gold": "etayee"},
                     "X1": {"gold": "non_etayee", "original_text": "right 0.8"},
                     "X2": {"gold": "partielle", "original_text": "right"},
                     "F3": {"gold": "partielle", "original_text": None}}}


def test_evaluate_counts_every_outcome():
    res = {"claims_final": [{"id": "F1", "text": "good"},               # kept
                            {"id": "F2", "text": "good two reworded"},  # good claim altered
                            {"id": "X1", "text": "right 0.8"},          # repaired
                            {"id": "F3", "text": "natural fixed"}],     # to review
           "removed": {"X2": "still_rejected"}, "unverified": [], "stats": {"prompt_tokens": 5, "completion_tokens": 2}}
    r = ep.evaluate(_evalset(), res)
    assert r["good_kept"] == ["F1"] and r["good_altered"] == ["F2"] and r["good_lost"] == []
    assert r["err_repaired"] == ["X1"] and r["err_review"] == ["F3"] and r["err_removed"] == ["X2"]
    assert r["err_remaining"] == []
    assert r["final_texts"] == {"F3": "natural fixed", "F2": "good two reworded"}


def test_writer_only_leaves_every_error_in_place():
    ev = _evalset()
    res = {"claims_final": ev["claims"], "removed": {}, "unverified": [], "stats": {}}
    r = ep.evaluate(ev, res)
    assert sorted(r["err_remaining"]) == ["F3", "X1", "X2"] and len(r["good_kept"]) == 2


def test_unverifiable_good_claim_counts_as_lost():
    ev = _evalset()
    res = {"claims_final": [c for c in ev["claims"] if c["id"] != "F1"], "removed": {}, "unverified": ["F1"], "stats": {}}
    assert ep.evaluate(ev, res)["good_lost"] == ["F1"]


def test_similar_ignores_case_and_punctuation_but_not_numbers():
    assert ep.similar("Accuracy rises to 0.66.", "accuracy rises to 0.66")
    assert not ep.similar("Accuracy rises to 0.76.", "accuracy rises to 0.66")


def test_a_changed_digit_is_never_considered_repaired():
    assert not ep.similar("Faithfulness of 0.870 on the benchmark.", "Faithfulness of 0.970 on the benchmark.")
    assert ep.similar("Faithfulness of 0.970 on the benchmark.", "faithfulness of 0.970 on the benchmark")