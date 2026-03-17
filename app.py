# app.py
import streamlit as st

from utils.kakao_parser import parse_kakao_chat, split_sessions
from models.hugging_face import load_emotion_classifier, load_sbert_model
from features.dominance import (
  calc_start_ratio, calc_end_ratio, calc_participation_ratio,
  calc_emotion_dominance, compute_dominance_features,
)
from features.dependence import (
  calc_reply_time_asymmetry, calc_double_text_ratio,
  calc_qa_sincerity, compute_dependence_index,
)
from visualize.charts import (
  create_radar_chart, create_timeline_chart,
  create_reply_time_chart, create_emotion_chart,
)

# ── 상수 ──────────────────────────────────────────────────────────────────
PAGE_TITLE = "💘 연애 권력 불균형 진단"
DEFAULT_SESSION_GAP = 30

# 가중치 프리셋: None이면 각 함수의 DEFAULT_WEIGHTS 사용
WEIGHT_PRESETS = {
  "기본": {"dominance": None, "dependence": None},
  "답장속도 중시": {
    "dominance": None,
    "dependence": {"reply_time_ratio": 0.55, "double_text_ratio": 0.25, "qa_sincerity_gap": 0.20},
  },
  "감정 중시": {
    "dominance": {
      "initiation_ratio": 0.10, "ending_ratio": 0.10,
      "message_count_ratio": 0.10, "char_count_ratio": 0.05,
      "joy_gap": 0.30, "negative_gap": 0.35,
    },
    "dependence": None,
  },
}

PROGRESS_STEPS = [
  ("📂", "데이터 준비 중"),
  ("🤖", "모델 로딩 중"),
  ("😊", "감정 분류 중"),
  ("🔢", "임베딩 계산 중"),
  ("📊", "지표 계산 중"),
  ("✨", "시각화 생성 중"),
]


# ── 상태 관리 ─────────────────────────────────────────────────────────────
def _init_state():
  defaults = {
    "phase": "upload",
    "df_parsed": None,
    "me": None,
    "analysis_result": None,
  }
  for key, val in defaults.items():
    if key not in st.session_state:
      st.session_state[key] = val


def _reset():
  st.session_state.phase = "upload"
  st.session_state.df_parsed = None
  st.session_state.me = None
  st.session_state.analysis_result = None


# ── Phase 1: 업로드 화면 ─────────────────────────────────────────────────
def render_upload():
  st.title(PAGE_TITLE)
  st.caption("카카오톡 대화를 AI로 분석해 연애 권력 불균형을 진단해드려요 🔍")
  st.divider()

  uploaded = st.file_uploader(
    "카카오톡 대화 파일을 업로드하세요 (.csv)",
    type=["csv"],
    help="카카오톡 PC → 대화방 상단 메뉴 → 대화 내보내기 → CSV 형식",
  )

  if uploaded is None:
    st.info("💡 카카오톡 PC 앱 → 대화방 → 우상단 메뉴 → 대화 내보내기 → CSV")
    return

  try:
    df = parse_kakao_chat(uploaded)
  except ValueError as e:
    st.error(f"❌ {e}")
    return

  st.session_state.df_parsed = df
  users = df["User"].unique().tolist()
  min_date = df["Date"].min().date()
  max_date = df["Date"].max().date()

  # 파일 요약
  col1, col2, col3 = st.columns(3)
  col1.metric("💬 총 메시지", f"{len(df):,}개")
  col2.metric("📅 첫 대화", str(min_date))
  col3.metric("📅 마지막 대화", str(max_date))

  st.divider()

  # 기본 설정
  col_a, col_b = st.columns(2)
  with col_a:
    start_date = st.date_input("분석 시작일", value=min_date, min_value=min_date, max_value=max_date)
    me = st.selectbox("나는 누구인가요? 👤", users)
  with col_b:
    end_date = st.date_input("분석 종료일", value=max_date, min_value=min_date, max_value=max_date)
    preset = st.selectbox("가중치 프리셋 ⚖️", list(WEIGHT_PRESETS.keys()))

  # 고급 설정
  with st.expander("⚙️ 고급 설정"):
    session_gap = st.slider(
      "세션 구분 간격 (분)",
      min_value=10, max_value=120, value=DEFAULT_SESSION_GAP, step=5,
      help="이 시간 이상 대화가 없으면 새로운 대화 세션으로 분류합니다",
    )

  if start_date > end_date:
    st.warning("⚠️ 시작일이 종료일보다 늦습니다.")
    return

  st.divider()

  if st.button("🔍 분석 시작", use_container_width=True, type="primary"):
    st.session_state.update({
      "phase": "loading",
      "me": me,
      "start_date": start_date,
      "end_date": end_date,
      "session_gap": session_gap,
      "preset": preset,
    })
    st.rerun()


# ── Phase 2: 분석 로딩 화면 ──────────────────────────────────────────────
def render_loading():
  st.title("⏳ 분석 중이에요...")
  st.caption("AI가 열심히 대화를 읽고 있어요. 잠시만 기다려 주세요 🤖")
  st.divider()

  df = st.session_state.df_parsed
  me = st.session_state.me
  start_date = st.session_state.start_date
  end_date = st.session_state.end_date
  session_gap = st.session_state.get("session_gap", DEFAULT_SESSION_GAP)
  preset_weights = WEIGHT_PRESETS.get(st.session_state.get("preset", "기본"), WEIGHT_PRESETS["기본"])

  progress_bar = st.progress(0)
  status = st.empty()

  def _step(i):
    emoji, label = PROGRESS_STEPS[i]
    status.markdown(f"### {emoji} {label}...")
    progress_bar.progress((i + 1) / len(PROGRESS_STEPS))

  try:
    # Step 1: 데이터 준비
    _step(0)
    df_filtered = df[
      (df["Date"].dt.date >= start_date) &
      (df["Date"].dt.date <= end_date)
    ].copy().reset_index(drop=True)

    if df_filtered.empty:
      st.error("❌ 선택한 기간에 메시지가 없습니다. 기간을 다시 확인해주세요.")
      if st.button("↩️ 돌아가기"):
        st.session_state.phase = "upload"
        st.rerun()
      return

    df_filtered = split_sessions(df_filtered, session_gap)
    partner = [u for u in df_filtered["User"].unique() if u != me][0]

    # Step 2: 모델 로딩
    _step(1)
    classifier = load_emotion_classifier()
    sbert_model = load_sbert_model()

    # Step 3: 감정 분류
    _step(2)
    emotion_result = calc_emotion_dominance(df_filtered, me, classifier)

    # Step 4: 임베딩
    _step(3)
    qa_gap = calc_qa_sincerity(df_filtered, me, sbert_model)

    # Step 5: 지표 계산
    _step(4)
    initiation = calc_start_ratio(df_filtered, me)
    ending = calc_end_ratio(df_filtered, me)
    participation = calc_participation_ratio(df_filtered, me)
    reply_time = calc_reply_time_asymmetry(df_filtered, me)
    double_text = calc_double_text_ratio(df_filtered, me)
    double_text_partner = calc_double_text_ratio(df_filtered, partner)

    dominance_metrics = {
      "initiation_ratio": initiation,
      "ending_ratio": ending,
      "message_count_ratio": participation["message_count_ratio"],
      "char_count_ratio": participation["char_count_ratio"],
      "joy_gap": emotion_result["joy_gap"],
      "negative_gap": emotion_result["negative_gap"],
    }
    dependence_metrics = {
      "reply_time_ratio": reply_time["ratio"],
      "double_text_ratio": double_text,
      "qa_sincerity_gap": qa_gap,
    }

    dominance_index = compute_dominance_features(dominance_metrics, preset_weights["dominance"])
    dependence_index = compute_dependence_index(dependence_metrics, preset_weights["dependence"])

    # Step 6: 시각화
    _step(5)
    fig_radar = create_radar_chart(dominance_metrics, dependence_metrics, me, partner)
    fig_timeline = create_timeline_chart(df_filtered, me)
    fig_reply = create_reply_time_chart(df_filtered, me)
    fig_emotion = create_emotion_chart(emotion_result, me, partner)

    st.session_state.analysis_result = {
      "me": me,
      "partner": partner,
      "df_filtered": df_filtered,
      "dominance_metrics": dominance_metrics,
      "dependence_metrics": dependence_metrics,
      "dominance_index": dominance_index,
      "dependence_index": dependence_index,
      "emotion_result": emotion_result,
      "reply_time": reply_time,
      "double_text": double_text,
      "double_text_partner": double_text_partner,
      "qa_gap": qa_gap,
      "participation": participation,
      "fig_radar": fig_radar,
      "fig_timeline": fig_timeline,
      "fig_reply": fig_reply,
      "fig_emotion": fig_emotion,
    }
    st.session_state.phase = "result"
    st.rerun()

  except Exception as e:
    st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
    if st.button("↩️ 처음으로 돌아가기"):
      _reset()
      st.rerun()


# ── Phase 3: 결과 화면 ───────────────────────────────────────────────────
def _interpret(value, me, partner):
  """0~1 지수를 사람이 읽기 쉬운 문구로 변환."""
  if value >= 0.65:
    return f"🔴 **{me}** 쪽이 우위예요"
  if value <= 0.35:
    return f"🔵 **{partner}** 쪽이 우위예요"
  return "🟢 균형적이에요"


def render_result():
  r = st.session_state.analysis_result
  me = r["me"]
  partner = r["partner"]

  st.title("📊 분석 결과")
  st.caption(f"**{me}** vs **{partner}** 대화 권력 불균형 리포트")
  st.divider()

  # 요약 카드 3열
  dom = r["dominance_index"]
  dep = r["dependence_index"]
  balance = 1 - abs(dom - dep)

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric("👑 지배성 지수", f"{dom:.2f}", help="0.5=균형, 1=내가 대화 주도")
    st.caption(_interpret(dom, me, partner))
  with col2:
    st.metric("💔 의존도 지수", f"{dep:.2f}", help="0.5=균형, 1=내가 더 의존적")
    st.caption(_interpret(dep, me, partner))
  with col3:
    st.metric("⚖️ 균형 점수", f"{balance:.2f}", help="1에 가까울수록 균형적인 관계")
    st.caption("🟢 균형적이에요" if balance >= 0.7 else "🔴 불균형이 감지됐어요")

  st.divider()

  # 레이더 차트
  st.subheader("🕸️ 대화 권력 지도")
  st.plotly_chart(r["fig_radar"], width="stretch")

  st.divider()

  # 4탭 결과
  tab1, tab2, tab3, tab4 = st.tabs(["📈 대화량", "😊 감정", "⏱️ 답장 패턴", "🙏 성의도"])

  with tab1:
    st.plotly_chart(r["fig_timeline"], width="stretch")
    p = r["participation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{me} 메시지 비율", f"{p['message_count_ratio']:.1%}")
    c2.metric(f"{partner} 메시지 비율", f"{1 - p['message_count_ratio']:.1%}")
    c3.metric(f"{me} 평균 길이", f"{p['avg_length_me']:.0f}자")
    c4.metric(f"{partner} 평균 길이", f"{p['avg_length_partner']:.0f}자")

  with tab2:
    st.plotly_chart(r["fig_emotion"], width="stretch")
    joy_gap = r["emotion_result"]["joy_gap"]
    neg_gap = r["emotion_result"]["negative_gap"]
    c1, c2 = st.columns(2)
    c1.metric("기쁨 감정 차이 (나 - 상대)", f"{joy_gap:+.2f}", help="양수 = 내가 더 긍정적")
    c2.metric("부정 감정 차이 (나 - 상대)", f"{neg_gap:+.2f}", help="양수 = 내가 부정 감정이 더 많음")

  with tab3:
    rt = r["reply_time"]
    me_reply_min = rt["partner_to_me_median_sec"] / 60
    partner_reply_min = rt["me_to_partner_median_sec"] / 60

    # ── 상단 평균 답장 시간 카드 ─────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
      st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ede7f6, #e8eaf6);
                    border-radius: 16px; padding: 24px; margin-bottom: 8px;">
          <p style="color: #7986cb; margin: 0 0 8px; font-size: 14px;">⏱ 평균 답장 시간</p>
          <p style="color: #3949ab; margin: 0 0 4px; font-size: 40px; font-weight: 700; line-height: 1;">
            {me_reply_min:.1f}분
          </p>
          <p style="color: #888; margin: 0; font-size: 13px;">{me}님의 답장 속도</p>
        </div>
      """, unsafe_allow_html=True)
    with c2:
      st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fce4ec, #fce8e8);
                    border-radius: 16px; padding: 24px; margin-bottom: 8px;">
          <p style="color: #e91e63; margin: 0 0 8px; font-size: 14px;">⏱ 평균 답장 시간</p>
          <p style="color: #c2185b; margin: 0 0 4px; font-size: 40px; font-weight: 700; line-height: 1;">
            {partner_reply_min:.1f}분
          </p>
          <p style="color: #888; margin: 0; font-size: 13px;">{partner}님의 답장 속도</p>
        </div>
      """, unsafe_allow_html=True)

    st.write("")

    # ── 박스 플롯 ────────────────────────────────────────────
    with st.container(border=True):
      st.markdown("**답장 시간 분포 분석**")
      st.caption("Box Plot으로 본 응답 시간의 편차")
      st.plotly_chart(r["fig_reply"], width="stretch")

      # 분석 인사이트
      faster = me if me_reply_min < partner_reply_min else partner
      slower = partner if faster == me else me
      ratio = partner_reply_min / me_reply_min if me_reply_min > 0 else 1.0
      with st.container():
        st.markdown("**분석 결과**")
        insights = [
          f"**{faster}**님은 대체로 빠르고 일관된 답장 패턴을 보입니다",
          f"**{slower}**님은 답장 시간의 편차가 크며, 상황에 따라 응답 속도가 달라집니다",
          f"상대가 답장까지 평균 {partner_reply_min:.1f}분 소요됩니다",
        ]
        for insight in insights:
          st.markdown(f"- {insight}")

    st.write("")

    # ── 하단 더블텍스트 + 대화 시작 빈도 카드 ────────────────
    initiation_me = r["dominance_metrics"]["initiation_ratio"]
    double_me = r["double_text"]
    double_partner = r["double_text_partner"]
    double_total = double_me + double_partner if (double_me + double_partner) > 0 else 1.0
    double_me_pct = double_me / double_total
    double_partner_pct = double_partner / double_total

    c3, c4 = st.columns(2)
    with c3:
      st.markdown(f"""
        <div style="background: white; border: 1px solid #e9ecef;
                    border-radius: 16px; padding: 24px;">
          <p style="margin: 0 0 16px; font-size: 16px; font-weight: 600;">⚡ 더블 텍스트 패턴</p>
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 14px;">{me}</span>
            <span style="color: #6366f1; font-weight: 700;">{double_me:.1%}</span>
          </div>
          <div style="background: #e9ecef; border-radius: 4px; height: 8px; margin-bottom: 12px;">
            <div style="background: #6366f1; width: {double_me_pct:.0%}; height: 8px; border-radius: 4px;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 14px;">{partner}</span>
            <span style="color: #f43f5e; font-weight: 700;">{double_partner:.1%}</span>
          </div>
          <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
            <div style="background: #f43f5e; width: {double_partner_pct:.0%}; height: 8px; border-radius: 4px;"></div>
          </div>
        </div>
      """, unsafe_allow_html=True)
    with c4:
      partner_initiation = 1 - initiation_me
      st.markdown(f"""
        <div style="background: white; border: 1px solid #e9ecef;
                    border-radius: 16px; padding: 24px;">
          <p style="margin: 0 0 16px; font-size: 16px; font-weight: 600;">📈 대화 시작 빈도</p>
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 14px;">{me}</span>
            <span style="color: #6366f1; font-weight: 700;">{initiation_me:.0%}</span>
          </div>
          <div style="background: #e9ecef; border-radius: 4px; height: 8px; margin-bottom: 12px;">
            <div style="background: #6366f1; width: {initiation_me:.0%}; height: 8px; border-radius: 4px;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 14px;">{partner}</span>
            <span style="color: #f43f5e; font-weight: 700;">{partner_initiation:.0%}</span>
          </div>
          <div style="background: #e9ecef; border-radius: 4px; height: 8px;">
            <div style="background: #f43f5e; width: {partner_initiation:.0%}; height: 8px; border-radius: 4px;"></div>
          </div>
        </div>
      """, unsafe_allow_html=True)

  with tab4:
    qa = r["qa_gap"]
    st.metric("QA 성의도 차이 (나 - 상대)", f"{qa:+.3f}")
    if qa > 0.1:
      st.success(f"💝 **{me}**가 질문에 더 성의 있게 답변하고 있어요. (의존도 신호)")
    elif qa < -0.1:
      st.warning(f"😅 **{partner}**가 질문에 더 성의 있게 답변하고 있어요.")
    else:
      st.info("💫 둘 다 비슷한 성의로 답변하고 있어요.")
    st.caption("코사인 유사도 기반 측정 | 질문-답변 쌍의 의미적 연관도를 측정합니다")

  st.divider()

  if st.button("🔄 재분석하기", use_container_width=True):
    _reset()
    st.rerun()


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
  st.set_page_config(
    page_title="연애 권력 불균형 진단",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="collapsed",
  )
  _init_state()

  phase = st.session_state.phase
  if phase == "upload":
    render_upload()
  elif phase == "loading":
    render_loading()
  elif phase == "result":
    render_result()


if __name__ == "__main__":
  main()
