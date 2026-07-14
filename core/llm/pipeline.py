# llm/pipeline.py
from core.llm.anonymize import anonymize_messages
from core.llm.retrieval import build_windows, retrieve_evidence, AXIS_QUERIES, WINDOW_SIZE
from core.llm.judge import judge
from core.llm.compare import compare_scores


def _build_summary(tier1_result: dict) -> dict:
    """LLM에 줄 Tier1 집계 요약(가벼운 dict)."""
    return {
        "dominance_score": tier1_result.get("dominance_index"),
        "dependence_score": tier1_result.get("dependence_index"),
        "metrics": tier1_result.get("dominance_metrics", {}),
        "emotion": tier1_result.get("emotion_result", {}),
    }


def run_llm_analysis(df_filtered, me: str, tier1_result: dict, encoder, config) -> dict:
    """Tier2 전체 실행: 익명화 → 윈도우 → RAG 검색 → 판단 → 비교."""
    df_anon = anonymize_messages(df_filtered, me)
    windows = build_windows(df_anon)
    top_k = max(1, config.max_messages // (len(AXIS_QUERIES) * WINDOW_SIZE))
    retrieved = retrieve_evidence(windows, encoder, top_k=top_k)
    summary = _build_summary(tier1_result)
    judgment = judge(config, summary, retrieved)
    comparison = compare_scores(
        tier1_result["dominance_index"],
        tier1_result["dependence_index"],
        judgment,
        config.agreement_threshold,
    )
    return {"judgment": judgment, "comparison": comparison, "retrieved": retrieved}
