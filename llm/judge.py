# llm/judge.py
import json
import re

_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    """모든 공백 제거 (비교 전용)."""
    return _WS.sub("", text)

from llm.client import call_structured
from llm.schema import LLMJudgment

_SYSTEM_PROMPT = """당신은 카카오톡 대화를 분석하는 연애 관계 분석 전문가입니다.
화자는 '나'와 '상대' 두 명입니다. 주어진 집계 통계와 검색된 근거 대화만 사용해
두 가지 축을 0~1 점수로 판단하세요.

[척도 정의]
- dominance(지배성): 0=상대 우위, 0.5=균형, 1=나 우위(내가 대화를 주도)
- dependence(의존도): 0=상대가 더 의존적, 0.5=균형, 1=내가 더 의존적

[규칙]
- evidence.quote는 반드시 '검색된 근거 대화'에 실제로 등장한 메시지 원문만 사용하세요. 창작 금지.
- 근거가 빈약하면 confidence를 낮추세요.
- report는 한국어 마크다운으로 관찰 + 부드러운 조언을 담으세요.
"""


def build_prompt(tier1_summary: dict, retrieved: dict) -> tuple[str, str]:
    """시스템/유저 프롬프트 문자열을 생성."""
    evidence_blocks = []
    for axis, windows in retrieved.items():
        joined = "\n---\n".join(w["text"] for w in windows)
        evidence_blocks.append(f"## {axis} 관련 검색 근거\n{joined}")

    user = (
        "다음은 규칙 기반(Tier1) 집계 결과입니다:\n"
        f"{json.dumps(tier1_summary, ensure_ascii=False, indent=2)}\n\n"
        "다음은 의미 검색으로 추출한 근거 대화입니다:\n"
        + "\n\n".join(evidence_blocks)
    )
    return _SYSTEM_PROMPT, user


def _corpus(retrieved: dict) -> str:
    return "\n".join(w["text"] for windows in retrieved.values() for w in windows)


def _clean_axis(axis, squashed_corpus: str):
    kept = [e for e in axis.evidence if _squash(e.quote) and _squash(e.quote) in squashed_corpus]
    return axis.model_copy(update={"evidence": kept})


def filter_hallucinated_evidence(judgment: LLMJudgment, retrieved: dict) -> LLMJudgment:
    """검색 근거에 존재하지 않는 인용을 제거한 새 판단 객체 반환."""
    squashed_corpus = _squash(_corpus(retrieved))
    return judgment.model_copy(update={
        "dominance": _clean_axis(judgment.dominance, squashed_corpus),
        "dependence": _clean_axis(judgment.dependence, squashed_corpus),
    })


def judge(config, tier1_summary: dict, retrieved: dict, client=None) -> LLMJudgment:
    """프롬프트 빌드 → LLM 호출 → 할루시네이션 필터 → 검증된 판단 반환."""
    system, user = build_prompt(tier1_summary, retrieved)
    raw = call_structured(config, system, user, LLMJudgment, client=client)
    return filter_hallucinated_evidence(raw, retrieved)
