import json

from src.agents import writer


def _papers_file(tmp_path):
    rows = [
        {"title": "Good paper", "abstract": "a" * 400, "url": "https://doi.org/10.1/x", "year": 2025, "relevance": 3},
        {"title": "Zenodo record", "abstract": "a" * 400, "url": "https://doi.org/10.5281/zenodo.1", "year": 2026, "relevance": 3},
        {"title": "Too short", "abstract": "short", "url": "https://doi.org/10.1/y", "year": 2025, "relevance": 3},
        {"title": "No judge note", "abstract": "a" * 400, "url": "https://doi.org/10.1/z", "year": 2025, "relevance": None},
        {"title": "Low note", "abstract": "a" * 400, "url": "https://doi.org/10.1/w", "year": 2025, "relevance": 1},
    ]
    f = tmp_path / "papers.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    return str(f)


def test_load_papers_filters_and_assigns_ids(tmp_path):
    papers = writer.load_papers(_papers_file(tmp_path))
    assert [p.title for p in papers] == ["Good paper", "No judge note"]
    assert [p.id for p in papers] == ["P1", "P2"]


def test_write_note_drops_unknown_paper_ids(tmp_path, monkeypatch):
    papers = writer.load_papers(_papers_file(tmp_path))
    fake = {"claims": [
        {"text": "Valid claim.", "paper_id": "P1"},
        {"text": "Cites a ghost.", "paper_id": "P99"},
        {"text": "   ", "paper_id": "P2"},
    ]}
    monkeypatch.setattr(writer.llm, "chat_json", lambda *a, **k: fake)
    claims = writer.write_note("topic", papers)
    assert len(claims) == 1 and claims[0].id == "C1" and claims[0].paper_id == "P1"


def test_write_note_drops_claims_with_foreign_script(tmp_path, monkeypatch):
    papers = writer.load_papers(_papers_file(tmp_path))
    fake = {"claims": [
        {"text": "A multi-step检索 pipeline reaches 0.66.", "paper_id": "P1"},
        {"text": "A multi-step retrieval pipeline reaches 0.66.", "paper_id": "P1"},
    ]}
    monkeypatch.setattr(writer.llm, "chat_json", lambda *a, **k: fake)
    claims = writer.write_note("topic", papers)
    assert [c.text for c in claims] == ["A multi-step retrieval pipeline reaches 0.66."]


def test_revise_returns_new_text_or_none_and_ignores_foreign_script(monkeypatch):
    from src.schemas import Claim, Paper
    papers = [Paper(id="P1", title="T", abstract="x" * 300)]
    items = [(Claim(id="C1", text="a", paper_id="P1"), "wrong number"),
             (Claim(id="C2", text="b", paper_id="P1"), "not in abstract"),
             (Claim(id="C3", text="c", paper_id="P1"), "wrong subject"),
             (Claim(id="C4", text="d", paper_id="P1"), "x")]
    fake = {"claims": [{"id": "C1", "text": "fixed"},
                       {"id": "C2", "text": None},
                       {"id": "C3", "text": "mixed 检索 text"},   # ignored: claim stays unchanged
                       {"id": "C9", "text": "unknown id"}]}        # ignored; C4 is simply not answered
    monkeypatch.setattr(writer.llm, "chat_json", lambda *a, **k: fake)
    assert writer.revise(items, papers) == {"C1": "fixed", "C2": None}


def test_plain_style_uses_the_baseline_prompt_without_strict_rules(monkeypatch):
    from src.schemas import Paper
    seen = {}

    def fake(system, user, **k):
        seen["system"] = system
        return {"claims": []}

    monkeypatch.setattr(writer.llm, "chat_json", fake)
    writer.write_note("t", [Paper(id="P1", title="T", abstract="x" * 300)], style="plain")
    assert "helpful assistant" in seen["system"] and "ONLY on them" not in seen["system"]
    writer.write_note("t", [Paper(id="P1", title="T", abstract="x" * 300)])
    assert "ONLY on them" in seen["system"]