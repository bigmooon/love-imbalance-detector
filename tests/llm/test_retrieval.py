# tests/llm/test_retrieval.py
import numpy as np
import pandas as pd
from core.llm.retrieval import build_windows, retrieve_for_axis, retrieve_evidence, AXIS_QUERIES


def _df(n_pairs=3):
    users, msgs = [], []
    for i in range(n_pairs):
        users += ["나", "상대"]
        msgs += [f"질문{i}", f"답변{i}"]
    return pd.DataFrame({
        "User": users,
        "Message": msgs,
        "Session_ID": [0] * len(users),
    })


def test_build_windows_chunks_messages():
    windows = build_windows(_df(2), window_size=2)
    assert len(windows) == 2  # 4 메시지 / 2
    assert "[나] 질문0" in windows[0]["text"]
    assert windows[0]["speakers"] == ["나", "상대"]
    assert windows[0]["session_id"] == 0


def test_retrieve_for_axis_picks_highest_cosine():
    windows = [
        {"text": "강한근거", "speakers": ["나"], "session_id": 0},
        {"text": "무관", "speakers": ["상대"], "session_id": 0},
    ]
    # stub 인코더: "강한근거"와 쿼리는 같은 방향, "무관"은 직교
    vecs = {"강한근거": [1.0, 0.0], "무관": [0.0, 1.0], "쿼리": [1.0, 0.0]}

    def encoder(texts):
        return np.array([vecs[t] for t in texts], dtype=float)

    result = retrieve_for_axis(windows, ["쿼리"], encoder, top_k=1)
    assert len(result) == 1
    assert result[0]["text"] == "강한근거"
    assert result[0]["sim"] > 0.9


def test_retrieve_evidence_covers_all_axes():
    windows = build_windows(_df(3), window_size=2)

    def encoder(texts):
        # 결정적 더미 벡터 (길이 기반)
        return np.array([[len(t), 1.0] for t in texts], dtype=float)

    out = retrieve_evidence(windows, encoder, top_k=2)
    assert set(out.keys()) == set(AXIS_QUERIES.keys())
    assert all(len(v) <= 2 for v in out.values())


def test_build_windows_empty():
    empty = pd.DataFrame({"User": [], "Message": [], "Session_ID": []})
    assert build_windows(empty) == []
