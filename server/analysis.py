"""UI 없는 분석 파이프라인. app.py render_loading()의 로직을 추출한 것."""
import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

from utils.kakao_parser import split_sessions
from features.presets import WEIGHT_PRESETS
from features.dominance import (
    calc_start_ratio, calc_end_ratio, calc_participation_ratio,
    calc_emotion_dominance, compute_dominance_features,
)
from features.dependence import (
    calc_reply_time_asymmetry, calc_double_text_ratio,
    calc_qa_sincerity, compute_dependence_index,
)
from llm.config import load_llm_config
from llm.client import LLMError
from llm.pipeline import run_llm_analysis
from models.hugging_face import encode_sentences

PROGRESS_LABELS = [
    "데이터 준비 중",
    "모델 로딩 중",
    "감정 분류 중",
    "임베딩 계산 중",
    "지표 계산 중",
    "리포트 구성 중",
    "LLM 심층 분석 중",
]
TOTAL_STEPS = len(PROGRESS_LABELS)


@dataclass(frozen=True)
class AnalysisOptions:
    me: str
    start_date: date
    end_date: date
    session_gap: int = 30
    preset: str = "기본"
    api_key: str | None = None


def run_analysis(
    df,
    opts: AnalysisOptions,
    progress_cb=lambda i, label: None,
    classifier=None,
    sbert_model=None,
) -> dict:
    """전체 분석 실행. 반환 dict는 app.py analysis_result와 동일 구조(figure 제외)."""

    def step(i):
        progress_cb(i, PROGRESS_LABELS[i])

    # Step 0: 데이터 준비
    step(0)
    df_filtered = df[
        (df["Date"].dt.date >= opts.start_date)
        & (df["Date"].dt.date <= opts.end_date)
    ].copy().reset_index(drop=True)
    if df_filtered.empty:
        raise ValueError("선택한 기간에 메시지가 없습니다. 기간을 다시 확인해주세요.")

    df_filtered = split_sessions(df_filtered, opts.session_gap)
    others = [u for u in df_filtered["User"].unique() if u != opts.me]
    if not others:
        raise ValueError("본인 외 대화 참여자가 없습니다.")
    partner = others[0]

    # Step 1: 모델 로딩 (미주입 시 실제 모델 — lazy import로 테스트 부담 제거)
    step(1)
    if classifier is None or sbert_model is None:
        from models.hugging_face import load_emotion_classifier, load_sbert_model
        classifier = classifier or load_emotion_classifier()
        sbert_model = sbert_model or load_sbert_model()

    # Step 2: 감정 분류
    step(2)
    emotion_result = calc_emotion_dominance(df_filtered, opts.me, classifier)

    # Step 3: 임베딩 (QA 성의도)
    step(3)
    qa_result = calc_qa_sincerity(df_filtered, opts.me, sbert_model)

    # Step 4: 지표 계산
    step(4)
    participation = calc_participation_ratio(df_filtered, opts.me)
    reply_time = calc_reply_time_asymmetry(df_filtered, opts.me)
    dominance_metrics = {
        "initiation_ratio": calc_start_ratio(df_filtered, opts.me),
        "ending_ratio": calc_end_ratio(df_filtered, opts.me),
        "message_count_ratio": participation["message_count_ratio"],
        "char_count_ratio": participation["char_count_ratio"],
        "joy_gap": emotion_result["joy_gap"],
        "negative_gap": emotion_result["negative_gap"],
    }
    dependence_metrics = {
        "reply_time_ratio": reply_time["ratio"],
        "double_text_ratio": calc_double_text_ratio(df_filtered, opts.me),
        "qa_sincerity_gap": qa_result["gap"],
    }
    weights = WEIGHT_PRESETS.get(opts.preset, WEIGHT_PRESETS["기본"])

    # Step 5: 결과 조립
    step(5)
    result = {
        "me": opts.me,
        "partner": partner,
        "df_filtered": df_filtered,
        "dominance_metrics": dominance_metrics,
        "dependence_metrics": dependence_metrics,
        "dominance_index": compute_dominance_features(dominance_metrics, weights["dominance"]),
        "dependence_index": compute_dependence_index(dependence_metrics, weights["dependence"]),
        "emotion_result": emotion_result,
        "reply_time": reply_time,
        "double_text": dependence_metrics["double_text_ratio"],
        "double_text_partner": calc_double_text_ratio(df_filtered, partner),
        "qa_sincerity": qa_result,
        "participation": participation,
    }

    # Step 6: LLM 심층 분석 (키 있을 때만, 실패해도 Tier1 보존 — app.py와 동일 정책)
    llm_result = None
    if opts.api_key:
        step(6)
        try:
            config = load_llm_config(api_key_override=opts.api_key)
            encoder = lambda texts: encode_sentences(texts, sbert_model)
            llm_result = run_llm_analysis(df_filtered, opts.me, result, encoder, config)
        except LLMError as e:
            logger.warning("LLM 분석 실패: %s", e)
            llm_result = {"error": str(e)}
        except Exception as e:  # 예기치 못한 오류도 Tier1은 보존
            logger.exception("LLM 분석 중 예상치 못한 오류")
            llm_result = {"error": f"예상치 못한 오류: {e}"}

    return {**result, "llm": llm_result}
