import json

from scripts import ablate_evidence_check as abl
from scripts import agreement as agr
from scripts import review_corrections as rc


def test_summarize_counts_hand_reviewed_repairs_and_damage():
    r = {"err_repaired": ["X1"], "err_review": ["F3", "F4"], "good_altered": ["F9", "F10"]}
    reviews = {"F3": {"label": "etayee"}, "F4": {"label": "partielle"},
               "F9": {"label": "etayee"}}
    s = rc.summarize(r, reviews)
    assert s["errors_repaired"] == 2 and s["rewrites_still_wrong"] == ["F4"]
    assert s["good_altered_ok"] == ["F9"] and s["pending"] == ["F10"] and s["good_altered_damaged"] == []


def test_label_without_check_undoes_only_the_mechanical_downgrade():
    down = {"label": "partielle", "reason": "[downgraded: evidence not found in abstract] ok"}
    real = {"label": "partielle", "reason": "the abstract does not mention a study"}
    assert abl.label_without_check(down) == "etayee" and abl.label_without_check(real) == "partielle"
    assert abl.label_without_check({"label": "non_etayee", "reason": "[downgraded x"}) == "non_etayee"


def test_ablation_metrics():
    gold = {"C1": "etayee", "C2": "etayee", "C3": "partielle"}
    m = abl.metrics(gold, {"C1": "partielle", "C2": "etayee", "C3": "partielle"})
    assert (m["false_alarms"], m["n_good"], m["detected"], m["n_err"]) == (1, 2, 1, 1)
    assert abs(m["accuracy"] - 2 / 3) < 1e-9


def test_agreement_kappa_perfect_and_chance_level():
    assert agr.kappa(["etayee", "partielle"], ["etayee", "partielle"]) == 1.0
    assert agr.kappa(["etayee", "etayee"], ["etayee", "etayee"]) == 1.0
    assert agr.kappa(["etayee", "partielle", "etayee", "partielle"],
                     ["etayee", "etayee", "partielle", "partielle"]) == 0.0


def test_agreement_compare_lists_disagreements_and_ignores_unlabelled():
    g1 = {"C1": ("etayee", "a"), "C2": ("partielle", "b"), "C3": ("etayee", "c")}
    g2 = {"C1": ("etayee", "a"), "C2": ("etayee", "b"), "C3": ("", "c")}
    ids, a, b, diff = agr.compare(g1, g2)
    assert ids == ["C1", "C2"] and diff == [("C2", "partielle", "etayee", "b")]


def test_old_review_is_ignored_when_the_rewrite_changed():
    reviews = {"F21": {"label": "etayee", "after": "Our method achieved superior performance."},
               "F20": {"label": "etayee", "after": "The paper proposes a pipeline."}}
    final = {"F21": "The authors' method achieved superior performance.", "F20": "The paper proposes a pipeline."}
    assert list(rc.valid_reviews(reviews, final)) == ["F20"]
    assert rc.valid_reviews(reviews, {}) == {}


def test_downgraded_lists_only_mechanical_downgrades_with_human_label():
    vs = [{"claim_id": "C1", "label": "partielle", "reason": "[downgraded: evidence not found] x"},
          {"claim_id": "C2", "label": "partielle", "reason": "real objection"},
          {"claim_id": "C3", "label": "partielle", "reason": "[downgraded: evidence not found] y"}]
    out = abl.downgraded(vs, {"C1": "etayee", "C3": "non_etayee"})
    assert out == [("C1", "etayee", "[downgraded: evidence not found] x"),
                   ("C3", "non_etayee", "[downgraded: evidence not found] y")]