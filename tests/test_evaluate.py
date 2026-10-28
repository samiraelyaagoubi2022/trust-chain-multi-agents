import json
import subprocess
import sys

from scripts.evaluate_verifier import kappa, load_pairs


def test_load_pairs_and_kappa(tmp_path):
    gold = {"annotations": [
        {"claim_id": "C1", "claim": "a", "human_label": "etayee"},
        {"claim_id": "C2", "claim": "b", "human_label": "partielle"},
        {"claim_id": "C3", "claim": "c", "human_label": ""}]}
    verd = {"verdicts": [{"claim_id": "C1", "label": "etayee"},
                         {"claim_id": "C2", "label": "partielle"},
                         {"claim_id": "C3", "label": "etayee"}]}
    g, v = tmp_path / "g.json", tmp_path / "v.json"
    g.write_text(json.dumps(gold)); v.write_text(json.dumps(verd))
    pairs, skipped, errors, missing = load_pairs(str(g), str(v))
    assert len(pairs) == 2 and skipped == ["C3"]
    assert kappa(pairs) == 1.0