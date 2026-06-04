# llm/compare.py
from llm.schema import LLMJudgment


def _one(tier1_score: float, llm_score: float, threshold: float) -> dict:
    delta = llm_score - tier1_score
    return {
        "tier1": float(tier1_score),
        "llm": float(llm_score),
        "delta": float(delta),
        "agree": abs(delta) <= threshold,
    }


def compare_scores(tier1_dominance: float, tier1_dependence: float,
                   judgment: LLMJudgment, threshold: float = 0.15) -> dict:
    """Tier1 점수와 LLM 점수를 축별로 비교한 dict 반환."""
    return {
        "dominance": _one(tier1_dominance, judgment.dominance.score, threshold),
        "dependence": _one(tier1_dependence, judgment.dependence.score, threshold),
        "agreement_threshold": threshold,
    }
