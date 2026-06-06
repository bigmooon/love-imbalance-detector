"""분석 결과 dict(app.py의 analysis_result 형태)를 ReportPayload로 변환."""
import numpy as np
import pandas as pd

from visualize.charts import RADAR_CATEGORIES, _normalize_for_radar
from server.schemas import (
    AxisComparison, BoxStats, EmotionPayload, EvidenceWindow, LLMPayload,
    PairRatio, Participation, QAPair, QASincerityPayload, RadarPayload,
    ReplyTimePayload, ReportPayload, TimelinePoint,
)

MAX_QA_PAIRS = 10


def _build_radar(dominance_metrics: dict, dependence_metrics: dict) -> RadarPayload:
    me_values = _normalize_for_radar(dominance_metrics, dependence_metrics)
    return RadarPayload(
        categories=list(RADAR_CATEGORIES),
        me=[float(v) for v in me_values],
        partner=[float(1 - v) for v in me_values],
    )


def _build_timeline(df: pd.DataFrame, me: str, partner: str) -> list[TimelinePoint]:
    df_week = df.copy()
    df_week["Week"] = df_week["Date"].dt.to_period("W").dt.start_time
    counts = df_week.groupby(["Week", "User"]).size().unstack(fill_value=0)
    return [
        TimelinePoint(
            week=week.date().isoformat(),
            me=int(row.get(me, 0)),
            partner=int(row.get(partner, 0)),
        )
        for week, row in counts.iterrows()
    ]


def _reply_minutes(df: pd.DataFrame, replier: str, original: str) -> pd.Series:
    """original이 말한 뒤 같은 세션에서 replier가 답하기까지 걸린 시간(분)."""
    prev_user = df["User"].shift(1)
    prev_date = df["Date"].shift(1)
    prev_session = df["Session_ID"].shift(1)
    is_reply = (
        (df["User"] == replier) & (prev_user == original)
        & (df["Session_ID"] == prev_session)
    )
    return (df["Date"] - prev_date).dt.total_seconds()[is_reply] / 60


def _box_stats(minutes: pd.Series) -> BoxStats:
    if len(minutes) == 0:
        return BoxStats(lo=0.0, q1=0.0, median=0.0, q3=0.0, hi=0.0)
    lo, q1, med, q3, hi = np.percentile(minutes, [5, 25, 50, 75, 95])
    return BoxStats(lo=float(lo), q1=float(q1), median=float(med), q3=float(q3), hi=float(hi))


def _build_reply_time(df: pd.DataFrame, me: str, partner: str, reply_time: dict) -> ReplyTimePayload:
    return ReplyTimePayload(
        me_median_sec=float(reply_time["partner_to_me_median_sec"]),
        partner_median_sec=float(reply_time["me_to_partner_median_sec"]),
        me_box=_box_stats(_reply_minutes(df, me, partner)),
        partner_box=_box_stats(_reply_minutes(df, partner, me)),
    )


def _build_llm(llm_result) -> tuple[LLMPayload | None, str | None]:
    if llm_result is None:
        return None, None
    if "error" in llm_result:
        return None, str(llm_result["error"])
    judgment = llm_result["judgment"]
    comparison = llm_result["comparison"]
    retrieved = llm_result["retrieved"]
    return LLMPayload(
        confidence=float(judgment.confidence),
        report=judgment.report,
        dominance=AxisComparison(**comparison["dominance"]),
        dependence=AxisComparison(**comparison["dependence"]),
        evidence={
            axis: [EvidenceWindow(text=w["text"], sim=float(w["sim"])) for w in windows[:4]]
            for axis, windows in retrieved.items()
        },
    ), None


def build_report_payload(result: dict) -> ReportPayload:
    me, partner = result["me"], result["partner"]
    df = result["df_filtered"]
    dom, dep = float(result["dominance_index"]), float(result["dependence_index"])
    qa = result["qa_sincerity"]
    llm_payload, llm_error = _build_llm(result.get("llm"))

    return ReportPayload(
        me=me,
        partner=partner,
        dominance_index=dom,
        dependence_index=dep,
        balance=float(1 - abs(dom - dep)),
        radar=_build_radar(result["dominance_metrics"], result["dependence_metrics"]),
        participation=Participation(**result["participation"]),
        timeline=_build_timeline(df, me, partner),
        emotion=EmotionPayload(**result["emotion_result"]),
        reply_time=_build_reply_time(df, me, partner, result["reply_time"]),
        double_text=PairRatio(me=float(result["double_text"]), partner=float(result["double_text_partner"])),
        initiation_ratio=float(result["dominance_metrics"]["initiation_ratio"]),
        qa_sincerity=QASincerityPayload(
            avg_sincerity=float(qa["avg_sincerity"]),
            my_sincerity=float(qa["my_sincerity"]),
            partner_sincerity=float(qa["partner_sincerity"]),
            pairs=[QAPair(**p) for p in qa["all_pairs"][:MAX_QA_PAIRS]],
        ),
        llm=llm_payload,
        llm_error=llm_error,
    )
