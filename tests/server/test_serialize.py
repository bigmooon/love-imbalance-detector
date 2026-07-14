import pandas as pd
import pytest

from server.serialize import build_report_payload


@pytest.fixture
def df_filtered():
    rows = [
        ("2025-01-06 10:00:00", "지언", "안녕 뭐해?", 0),
        ("2025-01-06 10:00:30", "민수", "일하지", 0),
        ("2025-01-06 10:05:00", "지언", "점심 먹었어?", 0),
        ("2025-01-06 10:20:00", "민수", "응", 0),
        ("2025-01-13 09:00:00", "지언", "주말에 뭐했어?", 1),
        ("2025-01-13 09:30:00", "민수", "그냥 쉬었어", 1),
    ]
    df = pd.DataFrame(rows, columns=["Date", "User", "Message", "Session_ID"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def analysis_result(df_filtered):
    return {
        "me": "지언", "partner": "민수", "df_filtered": df_filtered,
        "dominance_metrics": {
            "initiation_ratio": 1.0, "ending_ratio": 0.0,
            "message_count_ratio": 0.5, "char_count_ratio": 0.6,
            "joy_gap": 0.1, "negative_gap": -0.05,
        },
        "dependence_metrics": {
            "reply_time_ratio": 2.0, "double_text_ratio": 0.0, "qa_sincerity_gap": 0.1,
        },
        "dominance_index": 0.62, "dependence_index": 0.71,
        "emotion_result": {
            "me": {"joy": 0.5, "anger": 0.1, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "partner": {"joy": 0.4, "anger": 0.2, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "joy_gap": 0.1, "negative_gap": -0.05,
        },
        "reply_time": {"me_to_partner_median_sec": 465.0, "partner_to_me_median_sec": 30.0, "ratio": 0.06},
        "double_text": 0.33, "double_text_partner": 0.0,
        "qa_sincerity": {
            "gap": 0.1, "my_sincerity": 0.6, "partner_sincerity": 0.5, "avg_sincerity": 0.55,
            "all_pairs": [
                {"questioner": "지언", "question": "뭐해?", "answerer": "민수", "answer": "일하지", "score": 0.3},
            ] * 15,  # 15개 → 10개로 잘리는지 확인
        },
        "participation": {
            "message_count_ratio": 0.5, "char_count_ratio": 0.6,
            "avg_length_me": 8.0, "avg_length_partner": 4.0,
        },
        "llm": None,
    }


def test_basic_fields(analysis_result):
    payload = build_report_payload(analysis_result)
    assert payload.me == "지언"
    assert payload.partner == "민수"
    assert payload.balance == pytest.approx(1 - abs(0.62 - 0.71))


def test_radar_has_seven_axes(analysis_result):
    payload = build_report_payload(analysis_result)
    assert len(payload.radar.categories) == 7
    assert len(payload.radar.me) == 7
    # partner = 1 - me (app.py 레이더와 동일 규칙)
    assert payload.radar.partner[0] == pytest.approx(1 - payload.radar.me[0])


def test_timeline_weekly_counts(analysis_result):
    payload = build_report_payload(analysis_result)
    weeks = {p.week: p for p in payload.timeline}
    assert "2025-01-06" in weeks
    assert weeks["2025-01-06"].me == 2
    assert weeks["2025-01-06"].partner == 2
    assert weeks["2025-01-13"].me == 1


def test_reply_time_mapping(analysis_result):
    payload = build_report_payload(analysis_result)
    # me_median = partner_to_me (내가 답장하기까지)
    assert payload.reply_time.me_median_sec == 30.0
    assert payload.reply_time.partner_median_sec == 465.0
    assert payload.reply_time.me_box.median == pytest.approx(4.5)
    assert payload.reply_time.partner_box.median == pytest.approx(15.0)


def test_qa_pairs_capped_at_ten(analysis_result):
    payload = build_report_payload(analysis_result)
    assert len(payload.qa_sincerity.pairs) == 10


def test_llm_none_and_error(analysis_result):
    assert build_report_payload(analysis_result).llm is None
    analysis_result["llm"] = {"error": "boom"}
    payload = build_report_payload(analysis_result)
    assert payload.llm is None
    assert payload.llm_error == "boom"


def test_box_stats_empty_series_returns_zeros():
    from server.serialize import _box_stats
    import pandas as pd
    box = _box_stats(pd.Series([], dtype=float))
    assert box.lo == box.q1 == box.median == box.q3 == box.hi == 0.0


def test_llm_full(analysis_result):
    from core.llm.schema import LLMJudgment, AxisJudgment
    judgment = LLMJudgment(
        dominance=AxisJudgment(score=0.7, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=0.5, rationale="r", evidence=[]),
        report="## 리포트", confidence=0.8,
    )
    analysis_result["llm"] = {
        "judgment": judgment,
        "comparison": {
            "dominance": {"tier1": 0.62, "llm": 0.7, "delta": 0.08, "agree": True},
            "dependence": {"tier1": 0.71, "llm": 0.5, "delta": -0.21, "agree": False},
            "agreement_threshold": 0.15,
        },
        "retrieved": {
            "dominance": [{"text": "[나] 보고싶어", "speakers": ["나"], "session_id": 0, "sim": 0.7}],
            "dependence": [],
        },
    }
    payload = build_report_payload(analysis_result)
    assert payload.llm.confidence == 0.8
    assert payload.llm.dominance.agree is True
    assert payload.llm.evidence["dominance"][0].text == "[나] 보고싶어"
