import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_utils import is_question


def _get_partner(df, me):
  users = df["User"].unique().tolist()
  others = [u for u in users if u != me]
  if not others:
    raise ValueError(f"[DEPENDENCE] 본인 외 대화 참여자가 존재하지 않습니다.")
  return others[0]


def calc_reply_time_asymmetry(df, me):
  """
  상호 간의 답장 시간 비교
  - 답장은 같은 세션 내에서 화자가 바뀌는 시점으로 정의
  
  Returns:
    {
      "me_to_partner_median_sec": float,  # me가 말한 뒤 partner가 답장하기까지 중앙값(초)
      "partner_to_me_median_sec": float,  # partner가 말한 뒤 me가 답장하기까지 중앙값(초)
      "ratio": float,  # partner_to_me / me_to_partner (me가 더 빨리 답장할수록 < 1)
    }
  """
  partner = _get_partner(df, me)
  
  prev_user = df["User"].shift(1)
  prev_date = df["Date"].shift(1)
  prev_session = df["Session_ID"].shift(1)

  # 화자가 바뀐 시점 = 답장, 같은 세션 내에서만
  is_reply = (df["User"] != prev_user) & (df["Session_ID"] == prev_session)
  reply_seconds = (df["Date"] - prev_date).dt.total_seconds()

  # me가 말한 뒤 partner가 답장한 시간들
  is_me_to_partner = is_reply & (prev_user == me) & (df["User"] == partner)
  me_to_partner_times = reply_seconds[is_me_to_partner]

  # partner가 말한 뒤 me가 답장한 시간들
  is_partner_to_me = is_reply & (prev_user == partner) & (df["User"] == me)
  partner_to_me_times = reply_seconds[is_partner_to_me]

  me_to_partner_median = me_to_partner_times.median() if len(me_to_partner_times) > 0 else 0.0
  partner_to_me_median = partner_to_me_times.median() if len(partner_to_me_times) > 0 else 0.0

  # 0으로 나누는 경우 방지: 둘 다 0이면 균형(1.0)
  if me_to_partner_median == 0:
    ratio = 1.0
  else:
    ratio = partner_to_me_median / me_to_partner_median

  return {
    "me_to_partner_median_sec": float(me_to_partner_median),
    "partner_to_me_median_sec": float(partner_to_me_median),
    "ratio": float(ratio),
  }


def calc_double_text_ratio(df, me):
  """
  답장 없이 me가 연속 2회 이상 메시지를 보낸 횟수 / me 전체 메시지.

  높을수록 me가 일방적으로 대화를 이끄는 경향
  """
  total_me = (df["User"] == me).sum()
  if total_me == 0:
    return 0.0

  # 화자가 바뀌는 시점마다 새 블록 번호 부여
  is_speaker_change = df["User"] != df["User"].shift(1)
  block_id = is_speaker_change.cumsum()

  # me의 연속 발화 블록만 필터링
  df_with_block = df[["User"]].copy()
  df_with_block["block_id"] = block_id
  df_my_blocks = df_with_block[df_with_block["User"] == me]

  # 각 블록별 메시지 수에서 첫 1개를 제외한 나머지 = 더블 텍스트
  block_sizes = df_my_blocks.groupby("block_id").size()
  double_text_count = (block_sizes - 1).clip(lower=0).sum()

  return double_text_count / total_me


def calc_qa_sincerity(df, me, sbert_model):
  """
  질문-답변 쌍의 코사인 유사도 평균으로 성의도 측정.

  partner가 질문 → me가 답변한 쌍 vs me가 질문 → partner가 답변한 쌍을 각각 계산.
  유사도가 낮으면 질문과 무관한 답변 → 성의 없음.

  Returns:
    me의 답변 성의도 - partner의 답변 성의도 차이 (-1~1).
    양수면 me가 더 성의 있게 답변하는 쪽 (= me가 더 의존적).
  """
  from models.hugging_face import encode_sentences

  partner = _get_partner(df, me)

  my_answer_pairs, partner_answer_pairs = _extract_qa_pairs(df, me, partner)

  # 쌍이 너무 적으면 신뢰도 낮으므로 중립값 반환
  MIN_PAIRS = 5
  my_sincerity = _calc_pair_similarity(my_answer_pairs, sbert_model) if len(my_answer_pairs) >= MIN_PAIRS else 0.5
  partner_sincerity = _calc_pair_similarity(partner_answer_pairs, sbert_model) if len(partner_answer_pairs) >= MIN_PAIRS else 0.5

  return float(my_sincerity - partner_sincerity)


def _extract_qa_pairs(df, me, partner):
  """
  질문-답변 쌍을 추출.

  질문 바로 다음에 오는 상대방의 첫 메시지를 답변으로 간주한다.
  3개 메시지 이내에서 탐색.

  Returns:
    (my_answer_pairs, partner_answer_pairs)
  """
  messages = df["Message"].tolist()
  users = df["User"].tolist()

  my_answer_pairs = []
  partner_answer_pairs = []

  for i in range(len(messages) - 1):
    if not is_question(messages[i]):
      continue

    questioner = users[i]
    # 질문자와 다른 화자의 다음 메시지를 찾는다
    for j in range(i + 1, min(i + 4, len(messages))):
      if users[j] != questioner:
        pair = (messages[i], messages[j])
        if questioner == partner:
          my_answer_pairs.append(pair)
        else:
          partner_answer_pairs.append(pair)
        break

  return my_answer_pairs, partner_answer_pairs


def _calc_pair_similarity(pairs, sbert_model):
  """
  (question, answer) 쌍 리스트의 평균 코사인 유사도 계산
  """
  from models.hugging_face import encode_sentences

  if not pairs:
    return 0.5

  questions = [q for q, _ in pairs]
  answers = [a for _, a in pairs]

  q_vectors = encode_sentences(questions, sbert_model)
  a_vectors = encode_sentences(answers, sbert_model)

  # 쌍별 코사인 유사도: 대각선 요소만 필요 (i번 질문 ↔ i번 답변)
  similarities = []
  for q_vec, a_vec in zip(q_vectors, a_vectors):
    sim = cosine_similarity([q_vec], [a_vec])[0][0]
    similarities.append(sim)

  return float(np.mean(similarities))


def compute_dependence_index(metrics, weights=None):
  """
  각 지표를 가중합해 0~1 범위의 의존도 지수 반환

  0.5 = 균형, 1에 가까울수록 me가 더 의존적

  Args:
    metrics: {
      "reply_time_ratio": float,     # partner_to_me / me_to_partner (me가 빨리 답장할수록 < 1)
      "double_text_ratio": float,    # me의 더블텍스트 비율 (0~1)
      "qa_sincerity_gap": float,     # me 성의도 - partner 성의도 (-1~1, 양수=me가 더 성의)
    }

  Returns:
    0~1 범위의 종합 의존도 지수.
  """
  DEFAULT_WEIGHTS = {
    "reply_time_ratio": 0.35,
    "double_text_ratio": 0.35,
    "qa_sincerity_gap": 0.30,
  }
  w = weights or DEFAULT_WEIGHTS

  normalized = {}

  # reply_time_ratio: 1이 균형, <1이면 me가 더 빨리 답장 = 더 의존적
  # sigmoid 변환으로 0~1 매핑 → 반전: me가 빠를수록 의존도 높음
  ratio = metrics.get("reply_time_ratio", 1.0)
  normalized["reply_time_ratio"] = 1 - (ratio / (1 + ratio))

  # double_text_ratio: 이미 0~1이지만 보통 0.3 이하이므로 ×2 스케일
  normalized["double_text_ratio"] = min(1.0, metrics.get("double_text_ratio", 0.0) * 2)

  # qa_sincerity_gap: -1~1 → 0~1 (양수 = me가 더 성의 = 더 의존적)
  gap = metrics.get("qa_sincerity_gap", 0.0)
  normalized["qa_sincerity_gap"] = (gap + 1) / 2

  score = sum(normalized.get(k, 0.5) * v for k, v in w.items())
  total_weight = sum(w.values())

  return max(0.0, min(1.0, score / total_weight))