from datetime import date

import numpy as np
import pandas as pd
import pytest

from server.analysis import AnalysisOptions, run_analysis, PROGRESS_LABELS


@pytest.fixture
def raw_df():
    rows = []
    # 2개 세션, 질문-답변 쌍 충분히 포함
    base = pd.Timestamp("2025-01-06 10:00:00")
    msgs = [
        ("지언", "안녕 뭐해?"), ("민수", "일하지"), ("지언", "점심은 먹었어?"), ("민수", "응 먹었어"),
        ("지언", "오늘 저녁에 시간 돼?"), ("민수", "글쎄"), ("지언", "보고싶다"), ("민수", "나도"),
    ]
    for i, (user, msg) in enumerate(msgs):
        rows.append((base + pd.Timedelta(minutes=i), user, msg))
    df = pd.DataFrame(rows, columns=["Date", "User", "Message"])
    return df


def fake_classifier(batch):
    """기쁨 라벨 고정 반환 (HF pipeline 모사: 리스트의 리스트)."""
    return [[{"label": "기쁨", "score": 0.9}] for _ in batch]


class FakeSbert:
    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 8))


def test_run_analysis_returns_full_result(raw_df):
    opts = AnalysisOptions(
        me="지언", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
    )
    steps = []
    result = run_analysis(
        raw_df, opts,
        progress_cb=lambda i, label: steps.append((i, label)),
        classifier=fake_classifier, sbert_model=FakeSbert(),
    )
    assert result["me"] == "지언"
    assert result["partner"] == "민수"
    assert 0.0 <= result["dominance_index"] <= 1.0
    assert 0.0 <= result["dependence_index"] <= 1.0
    assert result["llm"] is None  # api_key 없음
    assert "df_filtered" in result
    # 진행 콜백이 순서대로 호출됨
    assert [i for i, _ in steps] == sorted([i for i, _ in steps])
    assert steps[0][1] == PROGRESS_LABELS[0]


def test_run_analysis_empty_range_raises(raw_df):
    opts = AnalysisOptions(
        me="지언", start_date=date(2030, 1, 1), end_date=date(2030, 12, 31),
    )
    with pytest.raises(ValueError, match="기간"):
        run_analysis(raw_df, opts, classifier=fake_classifier, sbert_model=FakeSbert())


def test_llm_error_preserves_tier1(raw_df, monkeypatch):
    """LLMError 발생 시 llm={"error": ...}, Tier1 결과는 보존."""
    import server.analysis as analysis_mod
    from core.llm.client import LLMError

    def boom(*args, **kwargs):
        raise LLMError("API 호출 실패")

    monkeypatch.setattr(analysis_mod, "run_llm_analysis", boom)
    opts = AnalysisOptions(
        me="지언", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
        api_key="sk-test",
    )
    result = run_analysis(raw_df, opts, classifier=fake_classifier, sbert_model=FakeSbert())
    assert result["llm"] == {"error": "API 호출 실패"}
    assert 0.0 <= result["dominance_index"] <= 1.0  # Tier1 보존
    assert result["partner"] == "민수"


def test_llm_unexpected_error_preserves_tier1(raw_df, monkeypatch):
    """예기치 못한 예외도 {"error": ...}로 변환되고 Tier1 보존."""
    import server.analysis as analysis_mod

    def boom(*args, **kwargs):
        raise RuntimeError("뭔가 잘못됨")

    monkeypatch.setattr(analysis_mod, "run_llm_analysis", boom)
    opts = AnalysisOptions(
        me="지언", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
        api_key="sk-test",
    )
    result = run_analysis(raw_df, opts, classifier=fake_classifier, sbert_model=FakeSbert())
    assert "error" in result["llm"]
    assert "뭔가 잘못됨" in result["llm"]["error"]
    assert 0.0 <= result["dependence_index"] <= 1.0  # Tier1 보존
