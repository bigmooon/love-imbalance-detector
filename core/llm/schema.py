# llm/schema.py
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    quote: str = Field(description="검색된 샘플에 실제 존재하는 메시지 원문")
    speaker: str = Field(description='"나" 또는 "상대"')
    reason: str = Field(description="이 인용이 판단을 뒷받침하는 이유")


class AxisJudgment(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0=상대 우위, 0.5=균형, 1=나 우위")
    rationale: str = Field(description="점수 판단 근거 요약")
    evidence: list[Evidence] = Field(default_factory=list)


class LLMJudgment(BaseModel):
    dominance: AxisJudgment
    dependence: AxisJudgment
    report: str = Field(description="마크다운 진단 리포트(관찰 + 조언)")
    confidence: float = Field(ge=0.0, le=1.0)
