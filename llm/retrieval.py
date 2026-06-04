# llm/retrieval.py
import numpy as np
import pandas as pd

# 축별 자연어 쿼리 시드 (RAG 검색의 질의문)
AXIS_QUERIES = {
    "dominance": [
        "한 사람이 먼저 연락하고 대화를 주도하는 장면",
        "감정적으로 우위에 있거나 주도권을 쥔 발화",
    ],
    "dependence": [
        "성의 없이 짧게 답하거나 무시하는 답변",
        "한쪽이 매달리거나 빠르게 답장하며 더 의존하는 장면",
    ],
}


def build_windows(df: pd.DataFrame, window_size: int = 4) -> list[dict]:
    """메시지를 window_size개씩 묶어 맥락 윈도우 리스트로 변환."""
    rows = list(zip(df["User"].tolist(), df["Message"].tolist(), df["Session_ID"].tolist()))
    windows = []
    for start in range(0, len(rows), window_size):
        chunk = rows[start:start + window_size]
        text = "\n".join(f"[{u}] {m}" for u, m, _ in chunk)
        windows.append({
            "text": text,
            "speakers": [u for u, _, _ in chunk],
            "session_id": int(chunk[0][2]),
        })
    return windows


def _cosine_to_query(win_vecs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """각 윈도우 벡터와 쿼리 벡터의 코사인 유사도."""
    win_norm = win_vecs / np.maximum(np.linalg.norm(win_vecs, axis=1, keepdims=True), 1e-9)
    q_norm = query_vec / max(float(np.linalg.norm(query_vec)), 1e-9)
    return win_norm @ q_norm


def retrieve_for_axis(windows: list[dict], axis_queries: list[str], encoder, top_k: int) -> list[dict]:
    """한 축의 쿼리들에 대해 가장 유사한 top_k 윈도우 반환."""
    if not windows:
        return []
    win_vecs = np.asarray(encoder([w["text"] for w in windows]), dtype=float)
    q_vecs = np.asarray(encoder(axis_queries), dtype=float)
    query_vec = q_vecs.mean(axis=0)
    sims = _cosine_to_query(win_vecs, query_vec)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{**windows[i], "sim": float(sims[i])} for i in top_idx]


def retrieve_evidence(windows: list[dict], encoder, queries: dict | None = None, top_k: int = 8) -> dict:
    """축별로 근거 윈도우를 검색해 dict로 반환."""
    queries = queries or AXIS_QUERIES
    return {axis: retrieve_for_axis(windows, qs, encoder, top_k) for axis, qs in queries.items()}
