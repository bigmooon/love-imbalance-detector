from datetime import datetime, timedelta

import pytest

from llm.config import LLMConfig
from eval.synth import (
  LEVELS,
  SCENARIOS,
  SESSION_COUNT,
  SessionTexts,
  TurnText,
  build_spec,
  build_turn_script,
  realized_stats,
  verbalize,
  assemble,
  generate_conversation,
)


def _cfg():
  return LLMConfig(api_key="sk-test", model="gpt-4o", max_messages=120,
                   request_timeout=10, max_retries=0, agreement_threshold=0.15)


class _FakeMessage:
  def __init__(self, parsed):
    self.parsed = parsed


class _FakeChoice:
  def __init__(self, parsed):
    self.message = _FakeMessage(parsed)


class _FakeCompletion:
  def __init__(self, parsed):
    self.choices = [_FakeChoice(parsed)]


class _QueueParseAPI:
  """세션별 호출마다 미리 준비한 parsed를 순서대로 반환."""
  def __init__(self, queue):
    self._queue = list(queue)
    self.calls = 0

  def parse(self, **kwargs):
    self.calls += 1
    return _FakeCompletion(self._queue.pop(0))


class _FakeClient:
  def __init__(self, queue):
    api = _QueueParseAPI(queue)
    self.api = api
    self.chat = type("Chat", (), {"completions": api})()


def _fake_client_for(script):
  """스크립트의 텍스트 턴을 세션 순서대로 채우는 fake 클라이언트."""
  queue = []
  for session in sorted({t["session"] for t in script}):
    turns = [
      TurnText(index=t["index"], text=f"메시지{t['index']}")
      for t in script
      if t["session"] == session and t["kind"] == "text"
    ]
    queue.append(SessionTexts(turns=turns))
  return _FakeClient(queue)


class TestBuildSpec:
  def test_gold_matches_levels(self):
    spec = build_spec(0.7, 0.3, "daily")
    assert spec["gold"] == {"dominance": 0.7, "dependence": 0.3}
    assert spec["scenario"] == "daily"

  def test_out_of_range_raises(self):
    with pytest.raises(ValueError):
      build_spec(1.5, 0.3, "daily")

  def test_unknown_scenario_raises(self):
    with pytest.raises(ValueError):
      build_spec(0.5, 0.5, "unknown_scenario")

  def test_behaviors_move_with_levels(self):
    low = build_spec(0.1, 0.1, "daily")["behaviors"]
    high = build_spec(0.9, 0.9, "daily")["behaviors"]
    assert high["initiation_ratio_a"] > low["initiation_ratio_a"]
    assert high["reply_delay_min_a"] < low["reply_delay_min_a"]  # 의존적일수록 빠른 답장


class TestTurnScript:
  def test_deterministic_for_same_seed(self):
    spec = build_spec(0.7, 0.3, "daily")
    assert build_turn_script(spec, seed=7) == build_turn_script(spec, seed=7)

  def test_different_seed_differs(self):
    spec = build_spec(0.7, 0.3, "daily")
    assert build_turn_script(spec, seed=1) != build_turn_script(spec, seed=2)

  def test_initiation_ratio_realized_exactly(self):
    spec = build_spec(0.7, 0.5, "daily")
    script = build_turn_script(spec, seed=3)
    stats = realized_stats(script)
    assert stats["initiation_ratio_a"] == pytest.approx(
      round(0.7 * SESSION_COUNT) / SESSION_COUNT
    )

  def test_message_share_close_to_target(self):
    spec = build_spec(0.9, 0.5, "daily")
    script = build_turn_script(spec, seed=3)
    stats = realized_stats(script)
    target = spec["behaviors"]["message_share_a"]
    assert abs(stats["message_share_a"] - target) < 0.05

  def test_message_share_not_contaminated_by_dependence_axis(self):
    # 회귀 방지: 연속톡(의존 축)이 발화량 비율(지배 축 gold)을 오염시키면 안 됨
    spec = build_spec(0.9, 0.1, "daily")
    script = build_turn_script(spec, seed=42)
    stats = realized_stats(script)
    target = spec["behaviors"]["message_share_a"]
    assert abs(stats["message_share_a"] - target) < 0.05

  def test_indexes_are_sequential(self):
    spec = build_spec(0.5, 0.5, "daily")
    script = build_turn_script(spec, seed=0)
    assert [t["index"] for t in script] == list(range(len(script)))

  def test_dependent_speaker_uses_more_nonverbal(self):
    spec = build_spec(0.5, 0.9, "daily")
    script = build_turn_script(spec, seed=5)
    stats = realized_stats(script)
    assert stats["nonverbal_rate_a"] > stats["nonverbal_rate_b"]


class TestAssemble:
  def _messages(self, l_dom=0.7, l_dep=0.3, seed=11):
    spec = build_spec(l_dom, l_dep, "daily")
    script = build_turn_script(spec, seed=seed)
    texts = {t["index"]: f"메시지{t['index']}" for t in script if t["kind"] == "text"}
    return spec, script, assemble(spec, script, texts, seed=seed)

  def test_timestamps_strictly_increasing(self):
    _, _, messages = self._messages()
    stamps = [datetime.strptime(m["datetime"], "%Y-%m-%d %H:%M:%S") for m in messages]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))

  def test_session_gap_over_30_minutes(self):
    _, _, messages = self._messages()
    for prev, cur in zip(messages, messages[1:]):
      if prev["session"] != cur["session"]:
        gap = (datetime.strptime(cur["datetime"], "%Y-%m-%d %H:%M:%S")
               - datetime.strptime(prev["datetime"], "%Y-%m-%d %H:%M:%S"))
        assert gap > timedelta(minutes=30)

  def test_emoticon_turns_use_kakao_placeholder(self):
    spec = build_spec(0.5, 0.9, "daily")
    script = build_turn_script(spec, seed=5)
    texts = {t["index"]: "채움" for t in script if t["kind"] == "text"}
    messages = assemble(spec, script, texts, seed=5)
    emoticons = [m for m in messages if m["type"] == "emoticon"]
    assert emoticons and all(m["text"] == "이모티콘" for m in emoticons)

  def test_missing_text_raises(self):
    spec = build_spec(0.5, 0.5, "daily")
    script = build_turn_script(spec, seed=0)
    with pytest.raises(ValueError):
      assemble(spec, script, {}, seed=0)


class TestVerbalize:
  def test_fills_all_text_turns(self):
    spec = build_spec(0.7, 0.3, "daily")
    script = build_turn_script(spec, seed=9)
    client = _fake_client_for(script)
    texts = verbalize(spec, script, _cfg(), client=client)
    text_indexes = {t["index"] for t in script if t["kind"] == "text"}
    assert set(texts.keys()) == text_indexes
    assert client.api.calls == SESSION_COUNT


class TestGenerateConversation:
  def test_returns_meta_and_messages(self):
    spec = build_spec(0.9, 0.1, "argument", variant=2)
    script = build_turn_script(spec, seed=42)
    client = _fake_client_for(script)
    convo = generate_conversation(spec, _cfg(), seed=42, client=client)
    assert convo["meta"]["gold"] == {"dominance": 0.9, "dependence": 0.1}
    assert convo["meta"]["realized"]["initiation_ratio_a"] == pytest.approx(0.9)
    assert len(convo["messages"]) == len(script)
