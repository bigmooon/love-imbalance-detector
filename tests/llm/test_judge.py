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


def test_filter_keeps_whitespace_variant_quote():
    """공백 차이가 있어도 실제 존재하는 인용은 보존되어야 한다."""
    # corpus window에는 공백 없이 "보고싶어서"가 들어 있음
    retrieved_ws = {
        "dominance": [{"text": "[나] 보고싶어서 연락했어\n[상대] 응", "speakers": ["나", "상대"], "session_id": 0, "sim": 0.9}],
        "dependence": [{"text": "[상대] ㅇㅇ", "speakers": ["상대"], "session_id": 1, "sim": 0.8}],
    }
    # LLM이 공백을 넣어 "보고 싶어서"로 반환한 경우 → 여전히 KEPT
    ws_variant = Evidence(quote="보고 싶어서", speaker="나", reason="연락 패턴")
    # 진짜 없는 인용 → REMOVED
    absent = Evidence(quote="존재하지않는말", speaker="나", reason="x")

    axis = AxisJudgment(score=0.7, rationale="r", evidence=[ws_variant, absent])
    judgment = LLMJudgment(dominance=axis, dependence=axis, report="rep", confidence=0.9)

    cleaned = filter_hallucinated_evidence(judgment, retrieved_ws)
    dom_quotes = [e.quote for e in cleaned.dominance.evidence]
    assert "보고 싶어서" in dom_quotes, "공백 변형 인용이 유지되어야 한다"
    assert "존재하지않는말" not in dom_quotes, "없는 인용은 제거되어야 한다"
