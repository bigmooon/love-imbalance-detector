# llm/anonymize.py
import pandas as pd


def anonymize_messages(df: pd.DataFrame, me: str) -> pd.DataFrame:
    """화자명을 '나'/'상대'로 치환한 새 DataFrame 반환 (입력 불변)."""
    out = df.copy()
    out["User"] = out["User"].apply(lambda u: "나" if u == me else "상대")
    return out
