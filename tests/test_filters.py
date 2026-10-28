from src.agents.researcher import Paper, looks_like_paper, rrf, norm


def _p(title="T", abstract="x" * 400, url="https://doi.org/10.18653/v1/x"):
    return Paper(source="openalex", title=title, abstract=abstract, url=url)


def test_zenodo_record_is_rejected():
    assert not looks_like_paper(_p(url="https://doi.org/10.5281/zenodo.21747271"))


def test_repository_abstract_is_rejected():
    assert not looks_like_paper(_p(abstract="Evaluation harness, logic-tree builder, cached trees..."))


def test_real_paper_is_kept():
    assert looks_like_paper(_p(abstract="We present MIRAGE, a QA dataset for RAG evaluation. " * 8))


def test_norm_merges_arxiv_and_openalex_titles():
    assert norm("RAGChecker: A Fine-Grained Framework") == norm("ragchecker a fine grained framework")


def test_rrf_prefers_item_ranked_first_twice():
    fused = rrf([[0, 1, 2], [0, 2, 1]], 3)
    assert fused[0] > fused[1] and fused[0] > fused[2]