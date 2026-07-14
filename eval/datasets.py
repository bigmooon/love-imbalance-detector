"""합성/공개 데이터 로더 — 표준 DataFrame(User, Message, Datetime, Session_ID, Message_Type)."""

import json
from pathlib import Path

import pandas as pd

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
COLUMNS = ["User", "Message", "Datetime", "Session_ID", "Message_Type"]


def save_conversation(convo, path):
  """합성 대화(meta+messages)를 UTF-8 JSON으로 저장한다."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(convo, ensure_ascii=False, indent=2), encoding="utf-8")


def load_synth_conversation(path):
  """합성 대화 JSON → (meta, 표준 DataFrame)."""
  try:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
  except (json.JSONDecodeError, OSError) as e:
    raise ValueError(f"합성 대화 파일을 읽을 수 없습니다: {path} ({e})")

  try:
    messages = raw["messages"]
    df = pd.DataFrame({
      "User": [m["user"] for m in messages],
      "Message": [m["text"] for m in messages],
      "Datetime": pd.to_datetime([m["datetime"] for m in messages], format=DATETIME_FORMAT),
      "Session_ID": [m["session"] for m in messages],
      "Message_Type": [m["type"] for m in messages],
    })
    return raw["meta"], df
  except KeyError as e:
    raise ValueError(f"필수 필드가 없습니다: {e} ({path})")


def load_synth_dir(dir_path):
  """디렉토리의 모든 합성 대화를 conversation_id 순으로 로드한다."""
  paths = sorted(Path(dir_path).glob("*.json"))
  if not paths:
    raise ValueError(f"합성 대화 JSON이 없습니다: {dir_path}")
  loaded = [load_synth_conversation(p) for p in paths]
  return sorted(loaded, key=lambda pair: pair[0]["conversation_id"])


def load_kote(split="test"):
  """KOTE(searle-j/kote)를 HF Hub에서 로드하고 라벨-매핑 일치를 자체 검증한다.

  네트워크가 필요하므로 지연 import. 라벨명이 eval/label_maps.py와 어긋나면
  조용히 틀린 벤치마크가 되는 대신 즉시 실패한다.
  """
  from datasets import load_dataset  # HuggingFace datasets (기존 의존성)
  from eval.label_maps import KOTE_TO_GROUP

  ds = load_dataset("searle-j/kote", split=split, trust_remote_code=True)
  names = ds.features["labels"].feature.names
  unknown = [n for n in names if n not in KOTE_TO_GROUP]
  if unknown:
    raise ValueError(f"매핑에 없는 KOTE 라벨: {unknown} — eval/label_maps.py를 갱신하세요")
  return ds
