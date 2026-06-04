# llm/client.py
from llm.config import LLMConfig


class LLMError(Exception):
    """LLM 호출 일반 오류."""


class LLMAuthError(LLMError):
    """API 키 누락/인증 오류."""


def _make_client(config: LLMConfig):
    from openai import OpenAI
    return OpenAI(api_key=config.api_key, timeout=config.request_timeout)


def call_structured(config: LLMConfig, system_prompt: str, user_prompt: str, schema_model, client=None):
    """OpenAI Structured Outputs로 schema_model 인스턴스를 반환. 실패 시 LLMError."""
    if not config.api_key:
        raise LLMAuthError("OPENAI_API_KEY가 설정되지 않았습니다.")

    client = client or _make_client(config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for _ in range(config.max_retries + 1):
        try:
            completion = client.chat.completions.parse(
                model=config.model,
                messages=messages,
                response_format=schema_model,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise LLMError("모델이 구조화 출력을 반환하지 않았습니다.")
            return parsed
        except LLMError:
            raise
        except Exception as e:  # 네트워크/레이트리밋 등 일시 오류 → 재시도
            last_err = e

    raise LLMError(f"LLM 호출에 실패했습니다: {last_err}")
