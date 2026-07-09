import numpy as np
import pytest

from eval.metrics import (
  macro_f1,
  mae,
  spearman,
  icc_2_1,
  sign_accuracy,
  swap_symmetry,
)


class TestMacroF1:
  def test_hand_computed(self):
    # class 0: P=1, R=0.5, F1=2/3 / class 1: P=2/3, R=1, F1=0.8
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 0.8) / 2)

  def test_perfect(self):
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

  def test_empty_raises(self):
    with pytest.raises(ValueError):
      macro_f1([], [])


class TestMae:
  def test_hand_computed(self):
    assert mae([0.5, 0.7], [0.4, 0.9]) == pytest.approx(0.15)

  def test_length_mismatch_raises(self):
    with pytest.raises(ValueError):
      mae([0.5], [0.4, 0.9])


class TestSpearman:
  def test_perfect_monotonic(self):
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

  def test_perfect_inverse(self):
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

  def test_too_short_raises(self):
    with pytest.raises(ValueError):
      spearman([1], [2])


class TestIcc21:
  def test_shrout_fleiss_1979(self):
    # Shrout & Fleiss (1979) 논문의 고전 예제 — ICC(2,1) ≈ 0.29
    ratings = np.array([
      [9, 2, 5, 8],
      [6, 1, 3, 2],
      [8, 4, 6, 8],
      [7, 1, 2, 6],
      [10, 5, 6, 9],
      [6, 2, 4, 7],
    ])
    assert icc_2_1(ratings) == pytest.approx(0.29, abs=0.01)

  def test_perfect_agreement(self):
    ratings = np.array([[1, 1], [2, 2], [3, 3]])
    assert icc_2_1(ratings) == pytest.approx(1.0)

  def test_requires_2d(self):
    with pytest.raises(ValueError):
      icc_2_1(np.array([1, 2, 3]))


class TestSignAccuracy:
  def test_hand_computed(self):
    # (+,+) 일치 / (-,+) 불일치 / (0,0) 일치 → 2/3
    y_true = [0.7, 0.3, 0.5]
    y_pred = [0.9, 0.6, 0.5]
    assert sign_accuracy(y_true, y_pred) == pytest.approx(2 / 3)

  def test_custom_center(self):
    assert sign_accuracy([1.0, -1.0], [0.5, -0.5], center=0.0) == pytest.approx(1.0)

  def test_empty_raises(self):
    with pytest.raises(ValueError):
      sign_accuracy([], [])


class TestSwapSymmetry:
  def test_perfect_symmetry_is_zero(self):
    # s + s_swap = 1 이면 편차 0
    assert swap_symmetry([0.7, 0.2], [0.3, 0.8]) == pytest.approx(0.0)

  def test_hand_computed(self):
    # |0.7+0.3-1|=0, |0.6+0.5-1|=0.1 → 평균 0.05
    assert swap_symmetry([0.7, 0.6], [0.3, 0.5]) == pytest.approx(0.05)

  def test_length_mismatch_raises(self):
    with pytest.raises(ValueError):
      swap_symmetry([0.5], [])
