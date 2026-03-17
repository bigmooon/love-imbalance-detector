# visualize/charts.py
import plotly.graph_objects as go

# 감정 그룹 한국어 레이블
EMOTION_GROUP_LABELS = {
  "anger": "분노",
  "sadness": "슬픔",
  "anxiety": "불안",
  "hurt": "상처",
  "embarrass": "당황",
  "joy": "기쁨",
}

EMOTION_COLORS = {
  "joy": "#FFD700",
  "anger": "#FF4B4B",
  "sadness": "#6495ED",
  "anxiety": "#FFA500",
  "hurt": "#9370DB",
  "embarrass": "#20B2AA",
}

RADAR_CATEGORIES = ["선톡 비율", "대화 종료", "메시지 비율", "글자 비율", "답장 속도", "더블텍스트", "QA 성의도"]


def create_radar_chart(dominance_metrics, dependence_metrics, speaker_a, speaker_b):
  """
  지배성·의존도 지표를 레이더 차트로 시각화.

  각 지표를 0~1로 정규화. 0.5 = 균형, 1에 가까울수록 speaker_a가 우위.
  """
  me_values = _normalize_for_radar(dominance_metrics, dependence_metrics)
  partner_values = [1 - v for v in me_values]

  # 닫힌 다각형을 만들기 위해 첫 값 반복
  categories = RADAR_CATEGORIES + [RADAR_CATEGORIES[0]]
  me_closed = me_values + [me_values[0]]
  partner_closed = partner_values + [partner_values[0]]

  fig = go.Figure()
  fig.add_trace(go.Scatterpolar(
    r=me_closed,
    theta=categories,
    fill="toself",
    name=speaker_a,
    opacity=0.7,
  ))
  fig.add_trace(go.Scatterpolar(
    r=partner_closed,
    theta=categories,
    fill="toself",
    name=speaker_b,
    opacity=0.7,
  ))
  fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title="대화 권력 지도",
  )
  return fig


def _normalize_for_radar(dominance_metrics, dependence_metrics):
  """
  레이더 차트 7개 축을 0~1로 정규화.

  gap 계수(-1~1)는 (x+1)/2로 변환, ratio 계수는 시그모이드 변환.
  """
  initiation = dominance_metrics.get("initiation_ratio", 0.5)
  ending = dominance_metrics.get("ending_ratio", 0.5)
  msg_ratio = dominance_metrics.get("message_count_ratio", 0.5)
  char_ratio = dominance_metrics.get("char_count_ratio", 0.5)

  # me가 빨리 답장할수록 ratio < 1 → 의존도 높음 → 레이더에서 높게 표시
  reply_ratio = dependence_metrics.get("reply_time_ratio", 1.0)
  reply_norm = 1 - (reply_ratio / (1 + reply_ratio))

  # 더블텍스트는 보통 0.3 이하이므로 ×2 스케일로 보정
  double_text = min(1.0, dependence_metrics.get("double_text_ratio", 0.0) * 2)

  # QA 성의도: -1~1 → 0~1 (양수 = me가 더 성의 = 더 의존적)
  qa_gap = dependence_metrics.get("qa_sincerity_gap", 0.0)
  qa_norm = (qa_gap + 1) / 2

  return [initiation, ending, msg_ratio, char_ratio, reply_norm, double_text, qa_norm]


def create_timeline_chart(df, speaker_a, metric="msg_count"):
  """
  주 단위 대화량 추이를 라인 차트로 시각화.
  """
  partner = _get_partner(df, speaker_a)

  df_week = df.copy()
  df_week["Week"] = df_week["Date"].dt.to_period("W").dt.start_time

  fig = go.Figure()
  for speaker in [speaker_a, partner]:
    df_speaker = df_week[df_week["User"] == speaker]
    weekly = df_speaker.groupby("Week").size().reset_index(name="count")
    fig.add_trace(go.Scatter(
      x=weekly["Week"],
      y=weekly["count"],
      mode="lines+markers",
      name=speaker,
    ))

  fig.update_layout(
    title="주간 대화량 추이",
    xaxis_title="주",
    yaxis_title="메시지 수",
  )
  return fig


def create_reply_time_chart(df, speaker_a):
  """
  A→B, B→A 답장 시간 분포를 수평 박스 플롯으로 시각화 (분 단위).

  같은 세션 내 화자가 바뀌는 시점을 답장으로 정의.
  """
  partner = _get_partner(df, speaker_a)

  prev_user = df["User"].shift(1)
  prev_date = df["Date"].shift(1)
  prev_session = df["Session_ID"].shift(1)

  is_reply = (df["User"] != prev_user) & (df["Session_ID"] == prev_session)
  reply_seconds = (df["Date"] - prev_date).dt.total_seconds()

  # me의 답장 시간: partner가 말한 뒤 me가 답장한 시간
  me_reply_minutes = reply_seconds[
    is_reply & (prev_user == partner) & (df["User"] == speaker_a)
  ] / 60

  # partner의 답장 시간: me가 말한 뒤 partner가 답장한 시간
  partner_reply_minutes = reply_seconds[
    is_reply & (prev_user == speaker_a) & (df["User"] == partner)
  ] / 60

  fig = go.Figure()
  fig.add_trace(go.Box(
    x=me_reply_minutes,
    name=speaker_a,
    orientation="h",
    marker_color="#6366f1",
    line_color="#6366f1",
    boxmean=True,
  ))
  fig.add_trace(go.Box(
    x=partner_reply_minutes,
    name=partner,
    orientation="h",
    marker_color="#f43f5e",
    line_color="#f43f5e",
    boxmean=True,
  ))
  fig.update_layout(
    title=dict(text="답장 시간 분포 분석"),
    xaxis_title="답장 시간 (분)",
    plot_bgcolor="#f8f9fa",
    paper_bgcolor="white",
    legend=dict(orientation="h", y=1.1),
  )
  fig.update_xaxes(showgrid=True, gridcolor="#e9ecef")
  return fig


def create_emotion_chart(emotion_results, speaker_a, speaker_b):
  """
  화자별 감정 분포를 stacked bar chart로 시각화.

  Args:
    emotion_results: calc_emotion_dominance() 반환값.
                     {"me": {group: ratio}, "partner": {group: ratio}, ...}
  """
  me_data = emotion_results.get("me", {})
  partner_data = emotion_results.get("partner", {})

  fig = go.Figure()
  for group, label in EMOTION_GROUP_LABELS.items():
    fig.add_trace(go.Bar(
      name=label,
      x=[speaker_a, speaker_b],
      y=[me_data.get(group, 0.0), partner_data.get(group, 0.0)],
      marker_color=EMOTION_COLORS.get(group, "#AAAAAA"),
    ))

  fig.update_layout(
    barmode="stack",
    title="화자별 감정 분포",
    yaxis_title="비율",
    yaxis=dict(range=[0, 1]),
  )
  return fig


def _get_partner(df, speaker_a):
  """df에서 speaker_a 외 다른 화자를 반환."""
  users = df["User"].unique().tolist()
  others = [u for u in users if u != speaker_a]
  if not others:
    raise ValueError("[CHARTS] speaker_a 외 대화 참여자가 없습니다.")
  return others[0]
