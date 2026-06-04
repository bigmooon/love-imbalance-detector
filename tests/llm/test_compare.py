# tests/llm/test_compare.py
from llm.compare import compare_scores
from llm.schema import AxisJudgment, LLMJudgment


def _judgment(dom, dep):
    return LLMJudgment(
        dominance=AxisJudgment(score=dom, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=dep, rationale="r", evidence=[]),
        report="rep", confidence=0.8,
    )


def test_agreement_within_threshold():
    out = compare_scores(0.60, 0.50, _judgment(0.66, 0.52), threshold=0.15)
    assert out["dominance"]["agree"] is True
    assert abs(out["dominance"]["delta"] - (0.66 - 0.60)) < 1e-9


def test_disagreement_beyond_threshold():
    out = compare_scores(0.30, 0.50, _judgment(0.80, 0.50), threshold=0.15)
    assert out["dominance"]["agree"] is False
    assert out["dependence"]["agree"] is True


def test_threshold_recorded():
    out = compare_scores(0.5, 0.5, _judgment(0.5, 0.5), threshold=0.2)
    assert out["agreement_threshold"] == 0.2
