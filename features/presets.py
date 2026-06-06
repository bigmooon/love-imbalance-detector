# features/presets.py
"""지배성/의존도 가중치 프리셋. None이면 각 compute 함수의 DEFAULT_WEIGHTS 사용."""

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
