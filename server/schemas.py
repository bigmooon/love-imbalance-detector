from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class UploadSummary(BaseModel):
    upload_id: str
    users: list[str]
    message_count: int
    first_date: date
    last_date: date


class AnalyzeRequest(BaseModel):
    upload_id: str
    me: str
    start_date: date
    end_date: date
    session_gap: int = Field(default=30, ge=10, le=120)
    preset: str = "기본"
    api_key: str | None = None


class RadarPayload(BaseModel):
    categories: list[str]
    me: list[float]
    partner: list[float]


class Participation(BaseModel):
    message_count_ratio: float
    char_count_ratio: float
    avg_length_me: float
    avg_length_partner: float


class TimelinePoint(BaseModel):
    week: str  # ISO 날짜 (주 시작일)
    me: int
    partner: int


class EmotionPayload(BaseModel):
    me: dict[str, float]
    partner: dict[str, float]
    joy_gap: float
    negative_gap: float


class BoxStats(BaseModel):
    """답장 시간 분포 (분 단위). lo/hi는 5/95 퍼센타일."""
    lo: float
    q1: float
    median: float
    q3: float
    hi: float


class ReplyTimePayload(BaseModel):
    me_median_sec: float       # partner가 말한 뒤 me가 답하기까지 중앙값
    partner_median_sec: float  # me가 말한 뒤 partner가 답하기까지 중앙값
    me_box: BoxStats
    partner_box: BoxStats


class PairRatio(BaseModel):
    me: float
    partner: float


class QAPair(BaseModel):
    questioner: str
    question: str
    answerer: str
    answer: str
    score: float


class QASincerityPayload(BaseModel):
    avg_sincerity: float
    my_sincerity: float
    partner_sincerity: float
    pairs: list[QAPair]


class AxisComparison(BaseModel):
    tier1: float
    llm: float
    delta: float
    agree: bool


class EvidenceWindow(BaseModel):
    text: str
    sim: float


class LLMPayload(BaseModel):
    confidence: float
    report: str
    dominance: AxisComparison
    dependence: AxisComparison
    evidence: dict[str, list[EvidenceWindow]]


class ReportPayload(BaseModel):
    me: str
    partner: str
    dominance_index: float
    dependence_index: float
    balance: float
    radar: RadarPayload
    participation: Participation
    timeline: list[TimelinePoint]
    emotion: EmotionPayload
    reply_time: ReplyTimePayload
    double_text: PairRatio
    initiation_ratio: float
    qa_sincerity: QASincerityPayload
    llm: LLMPayload | None = None
    llm_error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    step: int
    total_steps: int
    label: str
    result: ReportPayload | None = None
    error: str | None = None
