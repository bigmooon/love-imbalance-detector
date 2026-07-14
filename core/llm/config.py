# llm/config.py
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    model: str
    max_messages: int
    request_timeout: float
    max_retries: int
    agreement_threshold: float


def load_llm_config(api_key_override: str | None = None) -> LLMConfig:
    """환경변수(+선택 override)에서 LLM 설정을 불변 객체로 로드한다."""
    api_key = api_key_override or os.environ.get("OPENAI_API_KEY")
    return LLMConfig(
        api_key=api_key,
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        max_messages=int(os.environ.get("LLM_MAX_MESSAGES", "120")),
        request_timeout=float(os.environ.get("LLM_TIMEOUT", "60")),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "2")),
        agreement_threshold=float(os.environ.get("LLM_AGREE_THRESHOLD", "0.15")),
    )
