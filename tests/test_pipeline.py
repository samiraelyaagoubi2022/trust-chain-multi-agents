from src import pipeline
from src.schemas import Claim, Paper, Verdict

PAPERS = [Paper(id="P1", title="Paper one", abstract="a" * 300, year=2025, url="http://p1"),
          Paper(id="P2", title="Paper two", abstract="b" * 300)]


def _claims():
    return [Claim(id="C1", text="good claim", paper_id="P1"),
            Claim(id="C2", text="bad claim", paper_id="P1"),
            Claim(id="C3", text="other paper claim", paper_id="P2")]


def _verifier(rejected_texts, calls):
    """Fake verifier: a claim is rejected iff its text is in rejected_texts."""
    def fake(claims, papers, progress=False, **kw):
        calls.append([c.id for c in claims])
        return [Verdict(claim_id=c.id, label="partielle" if c.text in rejected_texts else "etayee",
                        reason="objection") for c in claims]
    return fake


def test_writer_only_changes_nothing():
    res = pipeline.run_mode("writer_only", _claims(), PAPERS)
    assert [c["id"] for c in res["claims_final"]] == ["C1", "C2", "C3"] and not res["removed"]


def test_verified_corrects_then_reverifies_only_the_changed_claim(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier({"bad claim"}, calls))
    monkeypatch.setattr(pipeline.writer, "revise", lambda items, papers, model=None: {"C2": "fixed claim"})
    res = pipeline.run_mode("verified", _claims(), PAPERS)
    assert calls == [["C1", "C2", "C3"], ["C2"]]                 # round 2 re-checks C2 only
    assert [c["text"] for c in res["claims_final"]] == ["good claim", "fixed claim", "other paper claim"]
    assert res["corrected"] == ["C2"] and not res["removed"]


def test_verified_deletes_when_the_writer_gives_up(monkeypatch):
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier({"bad claim"}, []))
    monkeypatch.setattr(pipeline.writer, "revise", lambda items, papers, model=None: {"C2": None})
    res = pipeline.run_mode("verified", _claims(), PAPERS)
    assert [c["id"] for c in res["claims_final"]] == ["C1", "C3"]
    assert res["removed"] == {"C2": "deleted_by_writer"}


def test_claims_still_rejected_after_the_last_round_are_dropped(monkeypatch):
    # the "fix" is still rejected: after max_rounds revisions the claim must not reach the note
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier({"bad claim", "still bad"}, []))
    monkeypatch.setattr(pipeline.writer, "revise", lambda items, papers, model=None: {"C2": "still bad"})
    res = pipeline.run_mode("verified", _claims(), PAPERS, max_rounds=2)
    assert [c["id"] for c in res["claims_final"]] == ["C1", "C3"]
    assert res["removed"] == {"C2": "still_rejected"} and len(res["rounds"]) == 3


def test_everything_valid_means_one_round_and_no_writer_call(monkeypatch):
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier(set(), []))

    def boom(*a, **k):
        raise AssertionError("revise must not be called")
    monkeypatch.setattr(pipeline.writer, "revise", boom)
    res = pipeline.run_mode("verified", _claims(), PAPERS)
    assert len(res["claims_final"]) == 3 and len(res["rounds"]) == 1


def test_claim_the_verifier_could_not_check_is_kept_out_of_the_note(monkeypatch):
    def fake(claims, papers, progress=False, **kw):
        return [Verdict(claim_id=c.id, label="erreur" if c.id == "C3" else "etayee") for c in claims]
    monkeypatch.setattr(pipeline.verifier, "verify_all", fake)
    res = pipeline.run_mode("verified", _claims(), PAPERS)
    assert [c["id"] for c in res["claims_final"]] == ["C1", "C2"] and res["unverified"] == ["C3"]


def test_self_review_applies_corrections_and_deletions(monkeypatch):
    monkeypatch.setattr(pipeline.writer, "self_review",
                        lambda claims, papers, model=None: {"C1": "good claim", "C2": "rewritten", "C3": None})
    res = pipeline.run_mode("self_review", _claims(), PAPERS)
    assert [c["text"] for c in res["claims_final"]] == ["good claim", "rewritten"]
    assert res["corrected"] == ["C2"] and res["removed"] == {"C3": "self_review"}


def test_assemble_note_groups_by_paper_and_lists_references():
    note = pipeline.assemble_note(_claims()[:1] + _claims()[2:], PAPERS, "RAG", verified=True)
    assert "## Paper one [P1]" in note and "- good claim [P1]" in note and "- other paper claim [P2]" in note
    assert "bad claim" not in note and "## References" in note and "- [P1] Paper one (2025) - http://p1" in note
    assert "checked against" in note


def test_assemble_note_skips_papers_without_claims():
    note = pipeline.assemble_note([_claims()[0]], PAPERS)
    assert "Paper two" not in note


def test_cached_verdicts_are_reused_and_not_verified_again(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier(set(), calls))
    key = lambda c: (c.text, c.paper_id, "v3", pipeline.VERIFIER_MODEL)
    cache = {key(c): {"claim_id": "OLD", "label": "etayee", "reason": "cached"} for c in _claims()[:2]}
    res = pipeline.run_mode("verified", _claims(), PAPERS, prompt="v3", cache=cache)
    assert calls == [["C3"]]                                   # only the uncached claim is verified
    assert res["rounds"][0]["reused"] == 2 and res["rounds"][0]["verified"] == 1
    assert res["verdicts"]["C1"]["claim_id"] == "C1"           # the id is the pipeline's, not the old file's


def test_cache_is_not_used_for_another_prompt_version(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.verifier, "verify_all", _verifier(set(), calls))
    cache = {(c.text, c.paper_id, "v1", pipeline.VERIFIER_MODEL): {"claim_id": "x", "label": "etayee"}
             for c in _claims()}
    pipeline.run_mode("verified", _claims(), PAPERS, prompt="v3", cache=cache)
    assert calls == [["C1", "C2", "C3"]]


def test_load_verdict_cache_maps_old_ids_to_claim_text(tmp_path):
    import json
    cf = tmp_path / "claims.json"
    cf.write_text(json.dumps({"claims": [{"id": "C1", "text": "t1", "paper_id": "P1"},
                                         {"id": "C2", "text": "t2", "paper_id": "P1"}]}), encoding="utf-8")
    vf = tmp_path / "verdicts.json"
    vf.write_text(json.dumps({"claims_file": str(cf), "prompt_version": "v3", "verifier_model": "m",
                              "verdicts": [{"claim_id": "C1", "label": "partielle", "reason": "r"},
                                           {"claim_id": "C2", "label": "erreur"}]}), encoding="utf-8")
    cache = pipeline.load_verdict_cache([str(vf)])
    assert list(cache) == [("t1", "P1", "v3", "m")] and cache[("t1", "P1", "v3", "m")]["label"] == "partielle"


def test_fresh_verdicts_are_reported_for_checkpointing(monkeypatch):
    def fake(claims, papers, progress=False, on_verdict=None, **kw):
        out = [Verdict(claim_id=c.id, label="etayee", reason="ok") for c in claims]
        for v in out:
            if on_verdict:
                on_verdict(v)
        return out

    monkeypatch.setattr(pipeline.verifier, "verify_all", fake)
    seen = {}
    pipeline.run_mode("verified", _claims(), PAPERS, prompt="v3", on_fresh=lambda k, v: seen.update({k: v}))
    assert len(seen) == 3
    assert ("good claim", "P1", "v3", pipeline.VERIFIER_MODEL) in seen


def test_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "ckpt.json")
    entries = {("t", "P1", "v3", "m"): {"claim_id": "C1", "label": "etayee"}}
    pipeline.save_checkpoint(path, entries)
    assert pipeline.load_checkpoint(path) == entries
    assert pipeline.load_checkpoint(str(tmp_path / "missing.json")) == {}