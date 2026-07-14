"""KOTE 44라벨 → 프로젝트 6대 감정 그룹 매핑.

매핑은 저자 판단 기반이며(논문 한계에 명시), 어긋난 라벨명은
eval/datasets.py의 load_kote()가 로드 시점에 즉시 검증한다.
"neutral"은 6그룹 어디에도 속하지 않아 벤치마크에서 제외하는 라벨.
"""

from core.models.emotion_labels import EMOTION_GROUPS

VALID_GROUPS = set(EMOTION_GROUPS.keys()) | {"neutral"}

KOTE_TO_GROUP = {
  # anger — 분노/경멸 계열
  "불평/불만": "anger",
  "지긋지긋": "anger",
  "화남/분노": "anger",
  "짜증": "anger",
  "역겨움/징그러움": "anger",
  "증오/혐오": "anger",
  "한심함": "anger",
  "어이없음": "anger",
  "우쭐댐/무시함": "anger",
  # sadness — 슬픔/상실 계열
  "슬픔": "sadness",
  "안타까움/실망": "sadness",
  "절망": "sadness",
  "패배/자기혐오": "sadness",
  "힘듦/지침": "sadness",
  "불쌍함/연민": "sadness",
  # anxiety — 불안/두려움 계열
  "공포/무서움": "anxiety",
  "불안/걱정": "anxiety",
  "의심/불신": "anxiety",
  "부담/안_내킴": "anxiety",
  # hurt — 상처 계열 (KOTE에서 직접 대응 라벨이 적음: 논문 한계에 기술)
  "서러움": "hurt",
  # embarrass — 당황/수치 계열
  "부끄러움": "embarrass",
  "당황/난처": "embarrass",
  "경악": "embarrass",
  "죄책감": "embarrass",
  # joy — 긍정 계열
  "환영/호의": "joy",
  "감동/감탄": "joy",
  "고마움": "joy",
  "존경": "joy",
  "기대감": "joy",
  "뿌듯함": "joy",
  "편안/쾌적": "joy",
  "신기함/관심": "joy",
  "아껴주는": "joy",
  "즐거움/신남": "joy",
  "흐뭇함(귀여움/예쁨)": "joy",
  "행복": "joy",
  "기쁨": "joy",
  "안심/신뢰": "joy",
  # neutral — 감정 극성이 불분명해 6그룹 벤치마크에서 제외
  "없음": "neutral",
  "깨달음": "neutral",
  "놀람": "neutral",
  "비장함": "neutral",
  "귀찮음": "neutral",
  "재미없음": "neutral",
}


def kote_to_group(label):
  """KOTE 라벨명을 6대 그룹(또는 neutral)으로 변환. 미지 라벨은 KeyError."""
  if label not in KOTE_TO_GROUP:
    raise KeyError(f"알 수 없는 KOTE 라벨: {label}")
  return KOTE_TO_GROUP[label]
