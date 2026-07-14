"""연구 평가용 공통 메트릭.

모든 함수는 순수 함수이며 입력을 변경하지 않는다.
- 분류 성능: macro_f1
- 점수 회수(회귀): mae, spearman, sign_accuracy
- LLM 판정 신뢰도: icc_2_1(반복 재현성), swap_symmetry(화자 스왑 대칭성)
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def _as_pair(y_true, y_pred, min_len=1):
  """두 시퀀스를 같은 길이의 numpy 배열 쌍으로 검증/변환."""
  a = np.asarray(y_true)
  b = np.asarray(y_pred)
  if len(a) != len(b):
    raise ValueError(f"길이가 다릅니다: {len(a)} != {len(b)}")
  if len(a) < min_len:
    raise ValueError(f"최소 {min_len}개 이상의 값이 필요합니다 (현재 {len(a)}개)")
  return a, b


def macro_f1(y_true, y_pred):
  """클래스별 F1의 단순 평균 (감정 분류 벤치마크용)."""
  a, b = _as_pair(y_true, y_pred)
  return float(f1_score(a, b, average="macro"))


def mae(y_true, y_pred):
  """평균 절대 오차 (합성 gold 점수 회수율용)."""
  a, b = _as_pair(y_true, y_pred)
  return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def spearman(y_true, y_pred):
  """스피어만 순위 상관계수 (점수 순위 보존 여부)."""
  a, b = _as_pair(y_true, y_pred, min_len=2)
  rho, _ = spearmanr(a, b)
  return float(rho)


def icc_2_1(ratings):
  """ICC(2,1): 이원 임의효과, 절대 일치, 단일 평가자.

  LLM 반복 판정의 재현성 측정용.

  Args:
    ratings: (n_targets, k_raters) 2차원 배열 — 행=대상(대화), 열=반복 판정 회차.
  """
  m = np.asarray(ratings, dtype=float)
  if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
    raise ValueError("ratings는 (대상 2개 이상 × 평가 2회 이상) 2차원 배열이어야 합니다")

  n, k = m.shape
  grand = m.mean()
  row_means = m.mean(axis=1)
  col_means = m.mean(axis=0)

  ss_rows = k * float(np.sum((row_means - grand) ** 2))
  ss_cols = n * float(np.sum((col_means - grand) ** 2))
  ss_total = float(np.sum((m - grand) ** 2))
  ss_error = ss_total - ss_rows - ss_cols

  ms_rows = ss_rows / (n - 1)
  ms_cols = ss_cols / (k - 1)
  ms_error = ss_error / ((n - 1) * (k - 1))

  denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
  if denom == 0:
    raise ValueError("분산이 0이라 ICC를 정의할 수 없습니다 (모든 값이 동일)")
  return float((ms_rows - ms_error) / denom)


def sign_accuracy(y_true, y_pred, center=0.5):
  """갑/을 방향(center 기준 부호) 일치 비율."""
  a, b = _as_pair(y_true, y_pred)
  a = a.astype(float)
  b = b.astype(float)
  return float(np.mean(np.sign(a - center) == np.sign(b - center)))


def swap_symmetry(scores, swapped_scores):
  """화자 스왑 대칭성 편차: mean |s + s_swap - 1|. 0이면 완벽한 대칭."""
  a, b = _as_pair(scores, swapped_scores)
  return float(np.mean(np.abs(a.astype(float) + b.astype(float) - 1.0)))
