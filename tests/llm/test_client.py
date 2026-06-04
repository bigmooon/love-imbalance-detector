# tests/llm/test_client.py
import pytest
from llm.config import LLMConfig
from llm.client import call_structured, LLMError, LLMAuthError
from llm.schema import LLMJudgment, AxisJudgment


def _cfg(key="sk-test", retries=1):
    return LLMConfig(api_key=key, model="m", max_messages=120,
                     request_timeout=10, max_retries=retries, agreement_threshold=0.15)


def _judgment():
    axis = AxisJudgment(score=0.5, rationale="r", evidence=[])
    return LLMJudgment(dominance=axis, dependence=axis, report="ok", confidence=0.5)


class _FakeMessage:
    def __init__(self, parsed):
        self.parsed = parsed


class _FakeChoice:
    def __init__(self, parsed):
        self.message = _FakeMessage(parsed)


class _FakeCompletion:
    def __init__(self, parsed):
        self.choices = [_FakeChoice(parsed)]


class _FakeParseAPI:
    def __init__(self, parsed=None, exc=None):
        self._parsed, self._exc, self.calls = parsed, exc, 0

    def parse(self, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return _FakeCompletion(self._parsed)


class _FakeClient:
    """client.chat.completions.parse(...) 경로를 모방 (stable path, openai>=2.0)."""
    def __init__(self, parsed=None, exc=None):
        self.api = _FakeParseAPI(parsed, exc)
        completions = self.api
        chat = type("Chat", (), {"completions": completions})()
        self.chat = chat


def test_missing_key_raises_auth_error():
    with pytest.raises(LLMAuthError):
        call_structured(_cfg(key=None), "sys", "user", LLMJudgment, client=_FakeClient())


def test_success_returns_parsed():
    client = _FakeClient(parsed=_judgment())
    result = call_structured(_cfg(), "sys", "user", LLMJudgment, client=client)
    assert result.report == "ok"


def test_retries_then_fails():
    client = _FakeClient(exc=RuntimeError("rate limit"))
    with pytest.raises(LLMError):
        call_structured(_cfg(retries=1), "sys", "user", LLMJudgment, client=client)
    assert client.api.calls == 2  # 최초 + 재시도 1회
