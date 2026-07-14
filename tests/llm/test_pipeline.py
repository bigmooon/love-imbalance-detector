# tests/llm/test_pipeline.py
import numpy as np
import pandas as pd
from core.llm.config import LLMConfig
from core.llm.schema import AxisJudgment, LLMJudgment
from core.llm import pipeline as pl


def _cfg():
    return LLMConfig(api_key="sk-test", model="m", max_messages=120,
                     request_timeout=10, max_retries=0, agreement_threshold=0.15)


def _df():
    return pd.DataFrame({
        "User": ["철수", "영희", "철수", "영희"],
        "Message": ["뭐해?", "응 왜", "보고싶어", "ㅇㅇ"],
        "Session_ID": [0, 0, 0, 0],
    })


def _tier1():
    return {"dominance_index": 0.6, "dependence_index": 0.55,
            "dominance_metrics": {}, "emotion_result": {"me": {}, "partner": {}}}


def test_run_llm_analysis_wires_everything(monkeypatch):
    fake_judgment = LLMJudgment(
        dominance=AxisJudgment(score=0.65, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=0.50, rationale="r", evidence=[]),
        report="## 리포트", confidence=0.8,
    )
    # judge를 가짜로 대체 (실제 OpenAI 호출 없음)
    monkeypatch.setattr(pl, "judge", lambda config, summary, retrieved, client=None: fake_judgment)

    def encoder(texts):
        return np.array([[len(t), 1.0] for t in texts], dtype=float)

    result = pl.run_llm_analysis(_df(), me="철수", tier1_result=_tier1(),
                                 encoder=encoder, config=_cfg())
    assert result["judgment"].report == "## 리포트"
    assert result["comparison"]["dominance"]["llm"] == 0.65
    assert set(result["retrieved"].keys()) == {"dominance", "dependence"}


def test_max_messages_caps_retrieved_windows(monkeypatch):
    """config.max_messages가 작으면 retrieved 윈도우 수가 그에 맞게 제한된다."""
    from core.llm.retrieval import AXIS_QUERIES, WINDOW_SIZE

    fake_judgment = LLMJudgment(
        dominance=AxisJudgment(score=0.5, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=0.5, rationale="r", evidence=[]),
        report="## r", confidence=0.7,
    )
    monkeypatch.setattr(pl, "judge", lambda config, summary, retrieved, client=None: fake_judgment)

    def encoder(texts):
        return np.array([[float(i), 1.0] for i, _ in enumerate(texts)], dtype=float)

    # 16 max_messages → top_k = 16 // (2 * 4) = 2
    small_cfg = LLMConfig(api_key="sk-test", model="m", max_messages=16,
                          request_timeout=10, max_retries=0, agreement_threshold=0.15)

    # 30 messages → 7–8 windows; enough that top_k=2 actually binds
    n_msgs = 30
    df_large = pd.DataFrame({
        "User": ["철수", "영희"] * (n_msgs // 2),
        "Message": [f"msg{i}" for i in range(n_msgs)],
        "Session_ID": [0] * n_msgs,
    })

    expected_top_k = max(1, small_cfg.max_messages // (len(AXIS_QUERIES) * WINDOW_SIZE))
    result = pl.run_llm_analysis(df_large, me="철수", tier1_result=_tier1(),
                                 encoder=encoder, config=small_cfg)

    for axis, windows in result["retrieved"].items():
        assert len(windows) <= expected_top_k, (
            f"axis={axis}: got {len(windows)} windows, expected at most {expected_top_k}"
        )
