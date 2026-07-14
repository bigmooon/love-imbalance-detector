# tests/llm/test_anonymize.py
import pandas as pd
from core.llm.anonymize import anonymize_messages


def _df():
    return pd.DataFrame({
        "User": ["철수", "영희", "철수"],
        "Message": ["안녕", "왜", "뭐해"],
        "Session_ID": [0, 0, 0],
    })


def test_maps_me_to_na_and_other_to_sangdae():
    out = anonymize_messages(_df(), me="철수")
    assert out["User"].tolist() == ["나", "상대", "나"]


def test_does_not_mutate_input():
    df = _df()
    anonymize_messages(df, me="철수")
    assert df["User"].tolist() == ["철수", "영희", "철수"]


def test_messages_preserved():
    out = anonymize_messages(_df(), me="철수")
    assert out["Message"].tolist() == ["안녕", "왜", "뭐해"]
