# tests/llm/test_schema.py
import pytest
from pydantic import ValidationError
from core.llm.schema import Evidence, AxisJudgment, LLMJudgment


def _valid_axis():
    return AxisJudgment(
        score=0.7,
        rationale="나가 대화를 주도함",
        evidence=[Evidence(quote="뭐해?", speaker="나", reason="선톡")],
    )


def test_llm_judgment_parses():
    j = LLMJudgment(
        dominance=_valid_axis(),
        dependence=_valid_axis(),
        report="## 리포트",
        confidence=0.8,
    )
    assert j.dominance.score == 0.7
    assert j.dominance.evidence[0].speaker == "나"


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AxisJudgment(score=1.5, rationale="x", evidence=[])


def test_empty_evidence_allowed():
    axis = AxisJudgment(score=0.5, rationale="근거 부족", evidence=[])
    assert axis.evidence == []
