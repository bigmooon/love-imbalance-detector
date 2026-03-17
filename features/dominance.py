import pandas as pd
from models.emotion_labels import EMOTION_GROUPS
from models.hugging_face import classify_emotions

def _get_partner(df, me):
  users = df["User"].unique().tolist()
  others = [u for u in users if u != me]
  if not others:
    raise ValueError(f"[DOMINANCE] 본인 외 대화 참여자가 존재하지 않습니다.")
  return others[0]


def calc_start_ratio(df, me):
  """
  세션별 선톡 비율 계산
  """
  df_start = df.groupby("Session_ID").first()
  total_sessions = len(df_start)
  if total_sessions == 0:
    return 0.0

  my_starts = (df_start["User"] == me).sum()
  return my_starts / total_sessions


def calc_end_ratio(df, me):
  """
  세션별 마지막 발화자 비율 계산
  """
  df_end = df.groupby("Session_ID").last()
  total_sessions = len(df_end)
  if total_sessions == 0:
    return 0.0

  my_ends = (df_end["User"] == me).sum()
  return my_ends / total_sessions


def calc_participation_ratio(df, me):
  """
  메시지 수, 총 글자 수, 평균 메시지 길이의 비율 계산
  
  Returns: 
    {
      "message_count_ratio": float,
      "char_count_ratio": float,
      "avg_length_me": float,
      "avg_length_partner": float
    }
  """
  partner = _get_partner(df, me)
  
  df_me = df[df["User"] == me]
  df_partner = df[df["User"] == partner]
  
  total_messages = len(df)
  if total_messages == 0:
    return {
      "message_count_ratio": 0.5,
      "char_count_ratio": 0.5,
      "avg_length_me": 0.0,
      "avg_length_partner": 0.0
    }
    
  char_lengths = df["Message"].str.len()
  char_me = char_lengths[df["User"] == me].sum()
  char_partner = char_lengths[df["User"] == partner].sum()
  char_total = char_me + char_partner
  return {
  "message_count_ratio": len(df_me) / total_messages,
  "char_count_ratio": char_me / char_total if char_total > 0 else 0.5,
  "avg_length_me": char_lengths[df["User"] == me].mean() if len(df_me) > 0 else 0.0,
  "avg_length_partner": char_lengths[df["User"] == partner].mean() if len(df_partner) > 0 else 0.0,
  }
  
  
def calc_emotion_dominance(df, me, classifier):
  partner = _get_partner(df, me)
  group_names = list(EMOTION_GROUPS.keys())
  
  texts = df["Message"].tolist()
  emotions = classify_emotions(texts, classifier)
  
  # 발화자별 감정 그룹 집계
  df_emo = df[["User"]].copy()
  df_emo["group"] = [e["group"] for e in emotions]
  
  me_counts = _calc_group_ratios(df_emo, me, group_names)
  partner_counts = _calc_group_ratios(df_emo, partner, group_names)
  
  # joy 제외 부정 감정 합산
  me_negative = sum(me_counts[g] for g in group_names if g != "joy")
  partner_negative = sum(partner_counts[g] for g in group_names if g != "joy")
  
  return {
    "me": me_counts,
    "partner": partner_counts,
    "joy_gap": me_counts.get("joy", 0) - partner_counts.get("joy", 0),
    "negative_gap": me_negative - partner_negative,
  }
  
  
def _calc_group_ratios(df_emo, user, group_names):
  """
  특정 사용자에 대해 감정 그룹별 비율 계산
  """
  df_user = df_emo[df_emo["User"] == user]
  total = len(df_user)
  if total == 0:
    return {group: 0.0 for group in group_names}
  
  counts = df_user["group"].value_counts()
  return {group: counts.get(group, 0) / total for group in group_names}


def compute_dominance_features(metrics, weights=None):
  """
  각 지표를 가중합해 0~1 사이의 지배성 점수로 변환
  
  0.5 = 균형, 0에 가까울 수록 me가 을 / 1에 가까울수록 me가 갑
  """
  DEFAULT_WEIGHTS = {
    "initiation_ratio": 0.20,
    "ending_ratio": 0.10,
    "message_count_ratio": 0.20,
    "char_count_ratio": 0.15,
    "joy_gap": 0.15,
    "negative_gap": 0.20,
  }
  w = weights or DEFAULT_WEIGHTS
 
  # gap 계수(-1~1)를 0~1 스케일로 변환
  normalized = {}
  for key, value in metrics.items():
    if key in ("joy_gap", "negative_gap"):
      normalized[key] = (value + 1) / 2
    else:
      normalized[key] = value
 
  score = sum(normalized.get(k, 0.5) * v for k, v in w.items())
  total_weight = sum(w.values())
 
  return max(0.0, min(1.0, score / total_weight))
