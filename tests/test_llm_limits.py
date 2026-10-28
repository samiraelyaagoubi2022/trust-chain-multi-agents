from src import llm
from src.agents import verifier
from src.schemas import Claim, Paper


def test_retry_after_parsing():
    assert llm._retry_after("Please try again in 1m23.4s.") == 83.4
    assert llm._retry_after("Please try again in 850ms") == 0.85
    assert llm._retry_after("try again in 2h3m") == 7380
    assert llm._retry_after("no hint here") is None


def test_daily_quota_detection():
    assert llm._is_daily_quota("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    assert not llm._is_daily_quota("Rate limit reached ... on tokens per minute (TPM): Limit 8000")


def test_verify_all_stops_after_three_errors_in_a_row(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Quota journalier Groq atteint")
    monkeypatch.setattr(verifier.llm, "chat_json", boom)
    papers = [Paper(id="P1", title="T", abstract="a" * 300)]
    claims = [Claim(id=f"C{i}", text="x", paper_id="P1") for i in range(10)]
    try:
        verifier.verify_all(claims, papers)
        assert False, "should have stopped"
    except RuntimeError as e:
        assert "3 erreurs d'affilée" in str(e)


def test_isolated_error_does_not_stop_the_run(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("one-off")
        return {"verdict": "supported", "evidence": "", "reason": "ok"}

    monkeypatch.setattr(verifier.llm, "chat_json", flaky)
    papers = [Paper(id="P1", title="T", abstract="a" * 300)]
    claims = [Claim(id=f"C{i}", text="x", paper_id="P1") for i in range(5)]
    out = verifier.verify_all(claims, papers)
    assert len(out) == 5 and [v.label for v in out].count("erreur") == 1


def test_quota_summary_is_short_and_shows_reset_delay():
    msg = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in "
           "organization `org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199100, "
           "Requested 3200. Please try again in 12m3.4s. Need more tokens? Upgrade at ...'}}")
    out = llm._quota_summary(msg)
    assert "openai/gpt-oss-120b" in out and "tokens per day (TPD)" in out and "12m3.4s" in out
    assert "org_x" not in out and len(out) < 200


def test_quota_summary_falls_back_when_format_is_unknown():
    assert llm._quota_summary("something unexpected") == "something unexpected"


# ---------- wait mode and resume ----------
import json
from types import SimpleNamespace

import httpx
from groq import RateLimitError


def _quota_error(delay="2s"):
    msg = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
           f"`on_demand` on tokens per day (TPD): Limit 200000, Used 199900, Requested 1100. "
           f"Please try again in {delay}. Need more tokens?")
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    return RateLimitError(msg, response=resp, body=None)


class _FakeClient:
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _quota_error()
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])


def test_daily_quota_aborts_by_default(monkeypatch):
    monkeypatch.setattr(llm, "_client", _FakeClient(fail_times=5))
    monkeypatch.setattr(llm, "WAIT_FOR_QUOTA", False)
    try:
        llm.chat_json("s", "u")
        assert False, "should abort"
    except RuntimeError as e:
        assert "Quota journalier" in str(e)


def test_wait_mode_sleeps_then_succeeds_without_using_up_retries(monkeypatch):
    fake = _FakeClient(fail_times=6)               # more failures than the 4 normal retries
    sleeps = []
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "WAIT_FOR_QUOTA", True)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    assert llm.chat_json("s", "u") == {"ok": True}
    assert fake.calls == 7 and len(sleeps) == 6 and all(3 <= x <= 3.5 for x in sleeps)  # 2s + 1s margin


def test_resume_skips_verdicts_already_done(monkeypatch):
    calls = []

    def fake(system, user, **k):
        calls.append(user)
        return {"verdict": "supported", "evidence": "", "reason": "ok"}

    monkeypatch.setattr(verifier.llm, "chat_json", fake)
    papers = [Paper(id="P1", title="T", abstract="a" * 300)]
    claims = [Claim(id=f"C{i}", text=f"claim {i}", paper_id="P1") for i in range(1, 4)]
    already = {"C1": verifier.Verdict(claim_id="C1", label="etayee")}
    saved = []
    out = verifier.verify_all(claims, papers, done=already, on_verdict=saved.append, check_evidence=False)
    assert len(calls) == 2 and [v.claim_id for v in out] == ["C1", "C2", "C3"]
    assert [v.claim_id for v in saved] == ["C2", "C3"]