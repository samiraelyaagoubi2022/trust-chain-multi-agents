from src.agents import verifier
from src.schemas import Claim, Paper

TITLE = "EvaRAG: Evaluating Advanced RAG Techniques"
ABSTRACT = ("This paper presents a benchmarking framework. The CRAG benchmark addresses the limitations "
            "of existing QA benchmarks in evaluating RAG systems. It provides a more comprehensive "
            "assessment of RAG performance. Accuracy rises from 0.40 to 0.66.")
SOURCE = f"{TITLE}. {ABSTRACT}"


def test_multi_fragment_evidence_with_ellipsis_is_accepted():
    ev = ("The CRAG benchmark addresses the limitations of existing QA benchmarks ... "
          "It provides a more comprehensive assessment of RAG performance")
    assert verifier.evidence_in_abstract(ev, SOURCE)


def test_evidence_given_as_list_is_accepted():
    assert verifier.evidence_in_abstract(["accuracy rises from 0.40 to 0.66"], SOURCE)


def test_name_that_only_appears_in_the_title_is_found():
    assert verifier.evidence_in_abstract("EvaRAG: Evaluating Advanced RAG Techniques", SOURCE)


def test_changed_number_is_never_tolerated_even_if_the_rest_matches():
    assert not verifier.evidence_in_abstract("Accuracy rises from 0.40 to 0.76", SOURCE)


def test_small_formatting_difference_is_tolerated():
    assert verifier.evidence_in_abstract("accuracy rises from 0.40 to 0.66 ,", SOURCE)


def test_invented_sentence_is_rejected():
    assert not verifier.evidence_in_abstract("The authors ran a user study with 200 participants", SOURCE)


def test_empty_evidence_is_rejected():
    assert not verifier.evidence_in_abstract("", SOURCE)


def test_prompt_v2_contains_checklist_and_v1_does_not():
    assert "SUBJECT" in verifier.PROMPTS["v2"] and "SUBJECT" not in verifier.PROMPTS["v1"]


def test_title_is_sent_to_the_model_and_evidence_from_title_counts(monkeypatch):
    seen = {}

    def fake(system, user, **k):
        seen["user"] = user
        return {"verdict": "supported", "evidence": "EvaRAG: Evaluating Advanced RAG Techniques", "reason": "ok"}

    monkeypatch.setattr(verifier.llm, "chat_json", fake)
    v = verifier.verify_claim(Claim(id="C1", text="EvaRAG evaluates RAG techniques.", paper_id="P1"),
                              Paper(id="P1", title=TITLE, abstract=ABSTRACT))
    assert "TITLE: " + TITLE in seen["user"] and v.label == "etayee"


def test_prompt_version_is_forwarded(monkeypatch):
    seen = {}

    def fake(system, user, **k):
        seen["system"] = system
        return {"verdict": "partial", "evidence": "", "reason": "x"}

    monkeypatch.setattr(verifier.llm, "chat_json", fake)
    verifier.verify_claim(Claim(id="C1", text="a", paper_id="P1"),
                          Paper(id="P1", title="T", abstract="b"), prompt="v2")
    assert "SUBJECT" in seen["system"]


def test_prompt_v3_clarifies_subject_and_modality_direction():
    v3 = verifier.PROMPTS["v3"]
    assert "this paper introduces X" in v3 and "weaker or more hedged" in v3
    assert "SUBJECT" in v3 and v3 != verifier.PROMPTS["v2"]
    # the example used in the prompt must not be one of the annotated development claims
    assert "interpreted to identify" not in v3