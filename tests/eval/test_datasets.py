import pandas as pd
import pytest

from eval.datasets import save_conversation, load_synth_conversation, load_synth_dir


def _convo(conversation_id="dom070_dep030_daily_v0"):
  return {
    "meta": {
      "conversation_id": conversation_id,
      "scenario": "daily",
      "gold": {"dominance": 0.7, "dependence": 0.3},
    },
    "messages": [
      {"user": "A", "text": "메시지0", "datetime": "2026-01-05 09:00:00", "session": 0, "type": "text"},
      {"user": "B", "text": "메시지1", "datetime": "2026-01-05 09:03:00", "session": 0, "type": "text"},
      {"user": "A", "text": "이모티콘", "datetime": "2026-01-05 18:00:00", "session": 1, "type": "emoticon"},
    ],
  }


class TestRoundTrip:
  def test_save_then_load(self, tmp_path):
    path = tmp_path / "convo.json"
    save_conversation(_convo(), path)
    meta, df = load_synth_conversation(path)

    assert meta["gold"] == {"dominance": 0.7, "dependence": 0.3}
    assert list(df.columns) == ["User", "Message", "Datetime", "Session_ID", "Message_Type"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["Datetime"])
    assert df["Session_ID"].tolist() == [0, 0, 1]
    assert df["Message_Type"].tolist() == ["text", "text", "emoticon"]

  def test_load_dir_sorted_by_id(self, tmp_path):
    save_conversation(_convo("b_convo"), tmp_path / "b.json")
    save_conversation(_convo("a_convo"), tmp_path / "a.json")
    loaded = load_synth_dir(tmp_path)
    assert [meta["conversation_id"] for meta, _ in loaded] == ["a_convo", "b_convo"]

  def test_empty_dir_raises(self, tmp_path):
    with pytest.raises(ValueError):
      load_synth_dir(tmp_path)

  def test_invalid_json_raises(self, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{잘못된 json", encoding="utf-8")
    with pytest.raises(ValueError):
      load_synth_conversation(path)
