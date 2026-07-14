from eval.label_maps import KOTE_TO_GROUP, VALID_GROUPS, kote_to_group

from core.models.emotion_labels import EMOTION_GROUPS


def test_covers_all_44_kote_labels():
  assert len(KOTE_TO_GROUP) == 44


def test_all_groups_valid():
  # 6대 그룹 + neutral(벤치마크 제외 라벨)만 허용
  assert VALID_GROUPS == set(EMOTION_GROUPS.keys()) | {"neutral"}
  assert set(KOTE_TO_GROUP.values()) <= VALID_GROUPS


def test_every_emotion_group_is_represented():
  # 6대 그룹 중 매핑이 비는 그룹이 있으면 macro-F1 비교가 불가능해짐
  mapped = set(KOTE_TO_GROUP.values()) - {"neutral"}
  assert mapped == set(EMOTION_GROUPS.keys())


def test_spot_checks():
  assert kote_to_group("기쁨") == "joy"
  assert kote_to_group("화남/분노") == "anger"
  assert kote_to_group("슬픔") == "sadness"
  assert kote_to_group("불안/걱정") == "anxiety"
  assert kote_to_group("없음") == "neutral"


def test_unknown_label_raises():
  import pytest
  with pytest.raises(KeyError):
    kote_to_group("존재하지않는라벨")
