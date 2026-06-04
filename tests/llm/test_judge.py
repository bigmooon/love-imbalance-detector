# tests/llm/test_judge.py
from llm.judge import build_prompt, filter_hallucinated_evidence
from llm.schema import Evidence, AxisJudgment, LLMJudgment


def _retrieved():
    return {
        "dominance": [{"text": "[나] 뭐해?\n[상대] 응", "speakers": ["나", "상대"], "session_id": 0, "sim": 0.9}],
        "dependence": [{"text": "[상대] ㅇㅇ", "speakers": ["상대"], "session_id": 1, "sim": 0.8}],
    }


def _summary():
    return {"dominance_score": 0.6, "dependence_score": 0.55, "metrics": {}, "emotion": {}}


def test_build_prompt_includes_evidence_and_scale():
    system, user = build_prompt(_summary(), _retrieved())
    assert "0.5" in system  # 척도 정의 명시
    assert "뭐해?" in user  # 검색 근거가 프롬프트에 포함


def test_filter_removes_hallucinated_quotes():
    real = Evidence(quote="뭐해?", speaker="나", reason="선톡")
    fake = Evidence(quote="존재하지않는인용", speaker="나", reason="x")
    axis = AxisJudgment(score=0.7, rationale="r", evidence=[real, fake])
    judgment = LLMJudgment(dominance=axis, dependence=axis, report="rep", confidence=0.9)

    cleaned = filter_hallucinated_evidence(judgment, _retrieved())
    dom_quotes = [e.quote for e in cleaned.dominance.evidence]
    assert "뭐해?" in dom_quotes
    assert "존재하지않는인용" not in dom_quotes


def test_filter_does_not_mutate_original():
    fake = Evidence(quote="없음", speaker="나", reason="x")
    axis = AxisJudgment(score=0.7, rationale="r", evidence=[fake])
    judgment = LLMJudgment(dominance=axis, dependence=axis, report="rep", confidence=0.9)
    filter_hallucinated_evidence(judgment, _retrieved())
    assert len(judgment.dominance.evidence) == 1  # 원본 보존
