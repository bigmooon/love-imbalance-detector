# tests/llm/test_config.py
from llm.config import LLMConfig, load_llm_config


def test_load_uses_override_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_llm_config(api_key_override="sk-test")
    assert isinstance(cfg, LLMConfig)
    assert cfg.api_key == "sk-test"


def test_load_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    cfg = load_llm_config()
    assert cfg.api_key == "sk-env"


def test_defaults_present(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_llm_config()
    assert cfg.api_key is None
    assert cfg.model
    assert cfg.max_messages > 0
    assert 0 < cfg.agreement_threshold < 1
