from src.agents import researcher as r
from src.agents.researcher import Paper, judge_structured, score_from_flags


def test_core_general_paper_scores_3():
    assert score_from_flags({"real_paper": True, "core_on_topic": True}) == 3


def test_domain_specific_is_rejected_even_if_core():
    assert score_from_flags({"real_paper": True, "core_on_topic": True, "single_domain": True}) == 1


def test_new_method_only_is_capped():
    assert score_from_flags({"core_on_topic": True, "new_method_only": True}) == 1


def test_survey_and_non_paper_score_0():
    assert score_from_flags({"core_on_topic": True, "survey_or_doc": True}) == 0
    assert score_from_flags({"core_on_topic": True, "real_paper": False}) == 0


def test_partly_on_topic_scores_2_and_string_booleans_work():
    assert score_from_flags({"partly_on_topic": "true"}) == 2


def _answer_all(prompt, temperature=0.0):
    """Faux LLM : répond pour chaque article numéroté [i] présent dans le prompt."""
    import re
    idx = [int(x) for x in re.findall(r"^\[(\d+)\] ", prompt, flags=re.M)]
    return {"papers": [{"i": i, "real_paper": True, "core_on_topic": True, "why": "ok"} for i in idx]}


def _papers(n):
    return [Paper(source="x", title=f"T{i}", abstract="a" * 400, url="") for i in range(n)]


def test_judge_structured_uses_rule_and_records_reason(monkeypatch):
    papers = _papers(2)
    fake = {"papers": [
        {"i": 0, "real_paper": True, "core_on_topic": True, "why": "general benchmark"},
        {"i": 1, "real_paper": True, "core_on_topic": True, "single_domain": True, "why": "soil science"},
    ]}
    monkeypatch.setattr(r, "llm_json", lambda *a, **k: fake)
    assert judge_structured("topic", papers) == [3, 1]
    assert papers[1].judge_why == "soil science"


def test_forgotten_papers_are_asked_again_not_silently_zeroed(monkeypatch):
    papers = _papers(3)
    calls = []

    def fake(prompt, temperature=0.0):
        calls.append(prompt)
        if len(calls) == 1:     # first call: the LLM only answers about paper 0
            return {"papers": [{"i": 0, "real_paper": True, "core_on_topic": True, "why": "a"}]}
        return _answer_all(prompt)     # second call: only the 2 forgotten ones are listed

    monkeypatch.setattr(r, "llm_json", fake)
    assert judge_structured("t", papers) == [3, 3, 3]
    assert len(calls) == 2 and "T0" not in calls[1]


def test_still_missing_after_retry_is_flagged(monkeypatch):
    papers = _papers(2)
    monkeypatch.setattr(r, "llm_json", lambda *a, **k: {"papers": []})
    assert judge_structured("t", papers) == [0, 0]
    assert papers[0].judge_why == "no answer from judge"


def test_large_lists_are_split_in_batches(monkeypatch):
    calls = []

    def fake(prompt, temperature=0.0):
        calls.append(prompt)
        return _answer_all(prompt)

    monkeypatch.setattr(r, "llm_json", fake)
    scores = judge_structured("t", _papers(25))
    assert scores == [3] * 25 and len(calls) == 3        # 10 + 10 + 5


def test_score_judge_is_batched_too(monkeypatch):
    calls = []

    def fake(prompt, temperature=0.0):
        import re
        calls.append(prompt)
        idx = [int(x) for x in re.findall(r"^\[(\d+)\] ", prompt, flags=re.M)]
        return {"scores": [{"i": i, "score": 2} for i in idx]}

    monkeypatch.setattr(r, "llm_json", fake)
    assert r.judge("t", _papers(12)) == [2] * 12 and len(calls) == 2


def test_judge_structured_failure_keeps_everything(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(r, "llm_json", boom)
    papers = _papers(1)
    assert judge_structured("t", papers) == [r.MIN_SCORE]