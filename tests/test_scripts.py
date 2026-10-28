import json
import random

from scripts import evaluate_verifier as ev
from scripts.make_synthetic_errors import add_unsupported_cause, perturb_direction, perturb_number


def test_number_is_changed():
    text = "Accuracy rises from 0.40 to 0.66 on the dataset."
    new_text, old, new = perturb_number(text, random.Random(0))
    assert old in text and new != old and new in new_text and new_text != text


def test_percent_stays_below_100():
    new_text, old, new = perturb_number("The model matches humans 91% of the time.", random.Random(0))
    assert new == "76" and "%" in new_text


def test_metric_names_are_not_treated_as_numbers():
    assert perturb_number("nDCG@10 and BLEU-4 improve for RAG systems.", random.Random(0)) is None


def test_direction_reversed():
    new_text, old, new = perturb_direction("SPLADE outperforms BM25 and DPR.")
    assert new_text == "SPLADE underperforms BM25 and DPR."


def test_direction_none_when_no_keyword():
    assert perturb_direction("RAGVUE is a framework for evaluation.") is None


def test_unsupported_cause_appended():
    new_text, _, clause = add_unsupported_cause("FAISS is fast.", random.Random(1))
    assert new_text.startswith("FAISS is fast,") and new_text.endswith(".") and clause


def test_family():
    assert ev.family("openai/gpt-oss-20b") == ev.family("openai/gpt-oss-120b")
    assert ev.family("qwen/qwen3.8-27b") != ev.family("openai/gpt-oss-120b")


def test_error_verdicts_are_excluded(tmp_path):
    gold = {"annotations": [
        {"claim_id": "C1", "claim": "a", "human_label": "etayee"},
        {"claim_id": "C2", "claim": "b", "human_label": "non_etayee"},
        {"claim_id": "C3", "claim": "c", "human_label": "partielle"},
        {"claim_id": "C4", "claim": "d", "human_label": ""},
    ]}
    verd = {"verdicts": [
        {"claim_id": "C1", "label": "etayee"},
        {"claim_id": "C2", "label": "erreur"},          # must NOT count as a detected error
        {"claim_id": "C3", "label": "partielle"},
    ]}
    g, v = tmp_path / "g.json", tmp_path / "v.json"
    g.write_text(json.dumps(gold)); v.write_text(json.dumps(verd))
    pairs, skipped, errors, missing = ev.load_pairs(str(g), str(v))
    assert [p[0] for p in pairs] == ["C1", "C3"]
    assert errors == ["C2"] and skipped == ["C4"] and missing == []


def test_missing_numbers_helper():
    from scripts.annotate import missing_numbers
    abstract = "Accuracy rises from 0.40 to 0.66, a 50% gain on 1,000 questions."
    assert missing_numbers("Accuracy rises from 0.40 to 0.66.", abstract) == []
    assert missing_numbers("Accuracy rises from 0.40 to 0.76.", abstract) == ["0.76"]
    assert missing_numbers("It reaches 1.736 ns latency.", abstract) == ["1.736"]