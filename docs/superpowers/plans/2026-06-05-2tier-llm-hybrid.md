# 2-Tier LLM 하이브리드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 BERT+규칙 기반(Tier1) 분석 위에 RAG 근거검색 + OpenAI LLM-as-judge(Tier2)를 더해, LLM과 BERT 점수를 비교하고 자연어 진단 리포트를 생성한다.

**Architecture:** Tier1은 전체 코퍼스를 싸게 분석하는 베이스라인이자 큐레이션 신호원으로 유지한다. 신규 `llm/` 패키지가 (1) SBERT 임베딩으로 축별 근거를 검색(RAG)하고 (2) 익명화된 근거+집계를 OpenAI에 1회 호출해 구조화 점수+리포트를 받고 (3) Tier1과 비교한다. Streamlit UI에 "AI 심층 분석" 섹션을 추가하되, 키가 없거나 호출이 실패해도 Tier1 결과는 항상 보존한다(graceful degradation).

**Tech Stack:** Python 3.12, OpenAI SDK(Structured Outputs/`beta.chat.completions.parse`), Pydantic v2, sentence-transformers(KR-SBERT, 기존 재사용), NumPy, Streamlit, pytest.

---

## File Structure

신규 (`llm/` 패키지):
- `llm/__init__.py` — 빈 패키지 마커
- `llm/config.py` — 환경설정 로딩 (`LLMConfig` frozen dataclass)
- `llm/schema.py` — Pydantic 구조화 출력 모델 (`Evidence`/`AxisJudgment`/`LLMJudgment`)
- `llm/anonymize.py` — 화자명 → `나`/`상대` 익명화 (새 DataFrame 반환)
- `llm/retrieval.py` — 메시지 윈도우 빌드 + 축별 코사인 근거 검색 (RAG 핵심)
- `llm/client.py` — OpenAI 래퍼 + 도메인 예외(`LLMError`/`LLMAuthError`) + 재시도
- `llm/judge.py` — 프롬프트 빌드 + 호출 + 할루시네이션 필터
- `llm/compare.py` — Tier1 vs Tier2 비교
- `llm/pipeline.py` — Tier2 오케스트레이터 (`run_llm_analysis`)
- `llm/ui.py` — Streamlit "AI 심층 분석" 섹션 렌더링

수정:
- `.gitignore` — `tests/` 추적 허용
- `pyproject.toml` / `requirements.txt` — `openai`, `pydantic`, `python-dotenv` 추가, packages.find 보완
- `app.py` — 사이드바 API 키 입력 + `render_loading`에서 Tier2 호출 + `render_result`에서 섹션 렌더

테스트 (`tests/llm/`):
- `tests/__init__.py`, `tests/llm/__init__.py`
- `tests/llm/test_config.py`, `test_schema.py`, `test_anonymize.py`, `test_retrieval.py`, `test_client.py`, `test_judge.py`, `test_compare.py`, `test_pipeline.py`

---

## Task 0: 프로젝트 셋업 (의존성 · gitignore · 패키지 골격)

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `llm/__init__.py`, `tests/__init__.py`, `tests/llm/__init__.py`

- [ ] **Step 1: `.gitignore`에서 `tests/` 추적 허용**

`.gitignore`에서 `tests/` 줄을 삭제한다. (테스트가 커밋되도록 — 포트폴리오 필수)
수정 후 `.gitignore` 전체:

```
__pycache__
data/
.venv/
06_huggingface_pipeline.egg-info/
.DS_Store
.env
```

(`.env` 추가: API 키 유출 방지)

- [ ] **Step 2: 의존성 추가**

Run:
```bash
uv add openai pydantic python-dotenv
uv add --dev pytest
```
Expected: `pyproject.toml`에 의존성 추가, lock 갱신. 실패 시 `pip install openai pydantic python-dotenv pytest`로 대체.

- [ ] **Step 3: `pyproject.toml`의 packages.find 보완**

`[tool.setuptools.packages.find]`의 include를 다음으로 교체:
```toml
[tool.setuptools.packages.find]
include = ["utils*", "models*", "features*", "visualize*", "llm*"]
```

또한 `requirements.txt` 끝에 다음 3줄을 추가:
```
openai>=1.40.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 4: 패키지 골격 생성**

```bash
mkdir -p llm tests/llm
touch llm/__init__.py tests/__init__.py tests/llm/__init__.py
```

- [ ] **Step 5: pytest 동작 확인**

Run: `uv run pytest -q` (또는 `python -m pytest -q`)
Expected: "no tests ran" (수집 0개, 에러 없음)

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt llm/__init__.py tests/
git commit -m "chore: scaffold llm package, add OpenAI deps, track tests"
```

---

## Task 1: `llm/config.py` — 환경설정

**Files:**
- Create: `llm/config.py`
- Test: `tests/llm/test_config.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.config'`

- [ ] **Step 3: 최소 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/config.py tests/llm/test_config.py
git commit -m "feat(llm): add LLMConfig environment loading"
```

---

## Task 2: `llm/schema.py` — 구조화 출력 모델

**Files:**
- Create: `llm/schema.py`
- Test: `tests/llm/test_schema.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_schema.py
import pytest
from pydantic import ValidationError
from llm.schema import Evidence, AxisJudgment, LLMJudgment


def _valid_axis():
    return AxisJudgment(
        score=0.7,
        rationale="나가 대화를 주도함",
        evidence=[Evidence(quote="뭐해?", speaker="나", reason="선톡")],
    )


def test_llm_judgment_parses():
    j = LLMJudgment(
        dominance=_valid_axis(),
        dependence=_valid_axis(),
        report="## 리포트",
        confidence=0.8,
    )
    assert j.dominance.score == 0.7
    assert j.dominance.evidence[0].speaker == "나"


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AxisJudgment(score=1.5, rationale="x", evidence=[])


def test_empty_evidence_allowed():
    axis = AxisJudgment(score=0.5, rationale="근거 부족", evidence=[])
    assert axis.evidence == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.schema'`

- [ ] **Step 3: 최소 구현**

```python
# llm/schema.py
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    quote: str = Field(description="검색된 샘플에 실제 존재하는 메시지 원문")
    speaker: str = Field(description='"나" 또는 "상대"')
    reason: str = Field(description="이 인용이 판단을 뒷받침하는 이유")


class AxisJudgment(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0=상대 우위, 0.5=균형, 1=나 우위")
    rationale: str = Field(description="점수 판단 근거 요약")
    evidence: list[Evidence] = Field(default_factory=list)


class LLMJudgment(BaseModel):
    dominance: AxisJudgment
    dependence: AxisJudgment
    report: str = Field(description="마크다운 진단 리포트(관찰 + 조언)")
    confidence: float = Field(ge=0.0, le=1.0)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_schema.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/schema.py tests/llm/test_schema.py
git commit -m "feat(llm): add Pydantic structured-output schema"
```

---

## Task 3: `llm/anonymize.py` — 화자 익명화

**Files:**
- Create: `llm/anonymize.py`
- Test: `tests/llm/test_anonymize.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_anonymize.py
import pandas as pd
from llm.anonymize import anonymize_messages


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_anonymize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.anonymize'`

- [ ] **Step 3: 최소 구현**

```python
# llm/anonymize.py
import pandas as pd


def anonymize_messages(df: pd.DataFrame, me: str) -> pd.DataFrame:
    """화자명을 '나'/'상대'로 치환한 새 DataFrame 반환 (입력 불변)."""
    out = df.copy()
    out["User"] = out["User"].apply(lambda u: "나" if u == me else "상대")
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_anonymize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/anonymize.py tests/llm/test_anonymize.py
git commit -m "feat(llm): add speaker anonymization for privacy"
```

---

## Task 4: `llm/retrieval.py` — RAG 근거 검색

**Files:**
- Create: `llm/retrieval.py`
- Test: `tests/llm/test_retrieval.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_retrieval.py
import numpy as np
import pandas as pd
from llm.retrieval import build_windows, retrieve_for_axis, retrieve_evidence, AXIS_QUERIES


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.retrieval'`

- [ ] **Step 3: 최소 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_retrieval.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/retrieval.py tests/llm/test_retrieval.py
git commit -m "feat(llm): add SBERT-based RAG evidence retrieval"
```

---

## Task 5: `llm/client.py` — OpenAI 래퍼

**Files:**
- Create: `llm/client.py`
- Test: `tests/llm/test_client.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
    """client.beta.chat.completions.parse(...) 경로를 모방."""
    def __init__(self, parsed=None, exc=None):
        self.api = _FakeParseAPI(parsed, exc)
        completions = self.api
        chat = type("Chat", (), {"completions": completions})()
        self.beta = type("Beta", (), {"chat": chat})()


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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.client'`

- [ ] **Step 3: 최소 구현**

```python
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
            completion = client.beta.chat.completions.parse(
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/client.py tests/llm/test_client.py
git commit -m "feat(llm): add OpenAI structured-output client with retries"
```

---

## Task 6: `llm/judge.py` — 프롬프트 + 호출 + 할루시네이션 필터

**Files:**
- Create: `llm/judge.py`
- Test: `tests/llm/test_judge.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_judge.py
from llm.judge import build_prompt, filter_hallucinated_evidence
from llm.schema import Evidence, AxisJudgment, LLMJudgment


def _retrieved():
    return {
        "dominance": [{"text": "[나] 뭐해?\n[상대] 응", "speakers": ["나", "상대"], "session_id": 0, "sim": 0.9}],
        "dependence": [{"text": "[상대] ㅇㅇ", "speakers": ["상대"], "session_id": 1, "sim": 0.8}],
    }


def _summary():
    return {"dominance_score": 0.6, "dependence_score": 0.55, "metrics": {}, "emotion": {}}


def test_build_prompt_includes_evidence_and_scale():
    system, user = build_prompt(_summary(), _retrieved())
    assert "0.5" in system  # 척도 정의 명시
    assert "뭐해?" in user  # 검색 근거가 프롬프트에 포함


def test_filter_removes_hallucinated_quotes():
    real = Evidence(quote="뭐해?", speaker="나", reason="선톡")
    fake = Evidence(quote="존재하지않는인용", speaker="나", reason="x")
    axis = AxisJudgment(score=0.7, rationale="r", evidence=[real, fake])
    judgment = LLMJudgment(dominance=axis, dependence=axis, report="rep", confidence=0.9)

    cleaned = filter_hallucinated_evidence(judgment, _retrieved())
    dom_quotes = [e.quote for e in cleaned.dominance.evidence]
    assert "뭐해?" in dom_quotes
    assert "존재하지않는인용" not in dom_quotes


def test_filter_does_not_mutate_original():
    fake = Evidence(quote="없음", speaker="나", reason="x")
    axis = AxisJudgment(score=0.7, rationale="r", evidence=[fake])
    judgment = LLMJudgment(dominance=axis, dependence=axis, report="rep", confidence=0.9)
    filter_hallucinated_evidence(judgment, _retrieved())
    assert len(judgment.dominance.evidence) == 1  # 원본 보존
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.judge'`

- [ ] **Step 3: 최소 구현**

```python
# llm/judge.py
import json

from llm.client import call_structured
from llm.schema import LLMJudgment

_SYSTEM_PROMPT = """당신은 카카오톡 대화를 분석하는 연애 관계 분석 전문가입니다.
화자는 '나'와 '상대' 두 명입니다. 주어진 집계 통계와 검색된 근거 대화만 사용해
두 가지 축을 0~1 점수로 판단하세요.

[척도 정의]
- dominance(지배성): 0=상대 우위, 0.5=균형, 1=나 우위(내가 대화를 주도)
- dependence(의존도): 0=상대가 더 의존적, 0.5=균형, 1=내가 더 의존적

[규칙]
- evidence.quote는 반드시 '검색된 근거 대화'에 실제로 등장한 메시지 원문만 사용하세요. 창작 금지.
- 근거가 빈약하면 confidence를 낮추세요.
- report는 한국어 마크다운으로 관찰 + 부드러운 조언을 담으세요.
"""


def build_prompt(tier1_summary: dict, retrieved: dict) -> tuple[str, str]:
    """시스템/유저 프롬프트 문자열을 생성."""
    evidence_blocks = []
    for axis, windows in retrieved.items():
        joined = "\n---\n".join(w["text"] for w in windows)
        evidence_blocks.append(f"## {axis} 관련 검색 근거\n{joined}")

    user = (
        "다음은 규칙 기반(Tier1) 집계 결과입니다:\n"
        f"{json.dumps(tier1_summary, ensure_ascii=False, indent=2)}\n\n"
        "다음은 의미 검색으로 추출한 근거 대화입니다:\n"
        + "\n\n".join(evidence_blocks)
    )
    return _SYSTEM_PROMPT, user


def _corpus(retrieved: dict) -> str:
    return "\n".join(w["text"] for windows in retrieved.values() for w in windows)


def _clean_axis(axis, corpus: str):
    kept = [e for e in axis.evidence if e.quote.strip() and e.quote.strip() in corpus]
    return axis.model_copy(update={"evidence": kept})


def filter_hallucinated_evidence(judgment: LLMJudgment, retrieved: dict) -> LLMJudgment:
    """검색 근거에 존재하지 않는 인용을 제거한 새 판단 객체 반환."""
    corpus = _corpus(retrieved)
    return judgment.model_copy(update={
        "dominance": _clean_axis(judgment.dominance, corpus),
        "dependence": _clean_axis(judgment.dependence, corpus),
    })


def judge(config, tier1_summary: dict, retrieved: dict, client=None) -> LLMJudgment:
    """프롬프트 빌드 → LLM 호출 → 할루시네이션 필터 → 검증된 판단 반환."""
    system, user = build_prompt(tier1_summary, retrieved)
    raw = call_structured(config, system, user, LLMJudgment, client=client)
    return filter_hallucinated_evidence(raw, retrieved)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_judge.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/judge.py tests/llm/test_judge.py
git commit -m "feat(llm): add judge prompt builder and hallucination filter"
```

---

## Task 7: `llm/compare.py` — Tier1 vs Tier2 비교

**Files:**
- Create: `llm/compare.py`
- Test: `tests/llm/test_compare.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_compare.py
from llm.compare import compare_scores
from llm.schema import AxisJudgment, LLMJudgment


def _judgment(dom, dep):
    return LLMJudgment(
        dominance=AxisJudgment(score=dom, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=dep, rationale="r", evidence=[]),
        report="rep", confidence=0.8,
    )


def test_agreement_within_threshold():
    out = compare_scores(0.60, 0.50, _judgment(0.66, 0.52), threshold=0.15)
    assert out["dominance"]["agree"] is True
    assert abs(out["dominance"]["delta"] - (0.66 - 0.60)) < 1e-9


def test_disagreement_beyond_threshold():
    out = compare_scores(0.30, 0.50, _judgment(0.80, 0.50), threshold=0.15)
    assert out["dominance"]["agree"] is False
    assert out["dependence"]["agree"] is True


def test_threshold_recorded():
    out = compare_scores(0.5, 0.5, _judgment(0.5, 0.5), threshold=0.2)
    assert out["agreement_threshold"] == 0.2
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.compare'`

- [ ] **Step 3: 최소 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_compare.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add llm/compare.py tests/llm/test_compare.py
git commit -m "feat(llm): add Tier1 vs Tier2 score comparison"
```

---

## Task 8: `llm/pipeline.py` — Tier2 오케스트레이터

**Files:**
- Create: `llm/pipeline.py`
- Test: `tests/llm/test_pipeline.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/llm/test_pipeline.py
import numpy as np
import pandas as pd
from llm.config import LLMConfig
from llm.schema import AxisJudgment, LLMJudgment
from llm import pipeline as pl


def _cfg():
    return LLMConfig(api_key="sk-test", model="m", max_messages=120,
                     request_timeout=10, max_retries=0, agreement_threshold=0.15)


def _df():
    return pd.DataFrame({
        "User": ["철수", "영희", "철수", "영희"],
        "Message": ["뭐해?", "응 왜", "보고싶어", "ㅇㅇ"],
        "Session_ID": [0, 0, 0, 0],
    })


def _tier1():
    return {"dominance_index": 0.6, "dependence_index": 0.55,
            "dominance_metrics": {}, "emotion_result": {"me": {}, "partner": {}}}


def test_run_llm_analysis_wires_everything(monkeypatch):
    fake_judgment = LLMJudgment(
        dominance=AxisJudgment(score=0.65, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=0.50, rationale="r", evidence=[]),
        report="## 리포트", confidence=0.8,
    )
    # judge를 가짜로 대체 (실제 OpenAI 호출 없음)
    monkeypatch.setattr(pl, "judge", lambda config, summary, retrieved, client=None: fake_judgment)

    def encoder(texts):
        return np.array([[len(t), 1.0] for t in texts], dtype=float)

    result = pl.run_llm_analysis(_df(), me="철수", tier1_result=_tier1(),
                                 encoder=encoder, config=_cfg())
    assert result["judgment"].report == "## 리포트"
    assert result["comparison"]["dominance"]["llm"] == 0.65
    assert set(result["retrieved"].keys()) == {"dominance", "dependence"}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest tests/llm/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.pipeline'`

- [ ] **Step 3: 최소 구현**

```python
# llm/pipeline.py
from llm.anonymize import anonymize_messages
from llm.retrieval import build_windows, retrieve_evidence
from llm.judge import judge
from llm.compare import compare_scores


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
    retrieved = retrieve_evidence(windows, encoder)
    summary = _build_summary(tier1_result)
    judgment = judge(config, summary, retrieved)
    comparison = compare_scores(
        tier1_result["dominance_index"],
        tier1_result["dependence_index"],
        judgment,
        config.agreement_threshold,
    )
    return {"judgment": judgment, "comparison": comparison, "retrieved": retrieved}
```

> 주의: `judge`를 `from llm.judge import judge`로 임포트해야 테스트의 `monkeypatch.setattr(pl, "judge", ...)`가 동작한다 (모듈 네임스페이스에 바인딩됨).

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/llm/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 전체 테스트 회귀 확인**

Run: `uv run pytest -q`
Expected: PASS (모든 llm 테스트 통과)

- [ ] **Step 6: Commit**

```bash
git add llm/pipeline.py tests/llm/test_pipeline.py
git commit -m "feat(llm): add Tier2 orchestration pipeline"
```

---

## Task 9: `llm/ui.py` — Streamlit "AI 심층 분석" 섹션

**Files:**
- Create: `llm/ui.py`

> 이 모듈은 Streamlit 렌더링이라 단위 테스트 대신 import-스모크로 검증한다.

- [ ] **Step 1: 구현 작성**

```python
# llm/ui.py
import streamlit as st


def _score_row(label: str, comp: dict):
    icon = "✅ AI도 동의" if comp["agree"] else "⚠️ 관점 차이"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{label} · 규칙(BERT)", f"{comp['tier1']:.2f}")
    c2.metric(f"{label} · LLM", f"{comp['llm']:.2f}", delta=f"{comp['delta']:+.2f}")
    c3.metric("일치 여부", icon)


def _evidence_cards(title: str, windows: list[dict]):
    st.markdown(f"**{title}**")
    for w in windows[:4]:
        st.markdown(
            f"<div style='background:#f7f7fb;border-radius:10px;padding:10px 14px;margin-bottom:8px;'>"
            f"<span style='color:#888;font-size:12px;'>유사도 {w['sim']:.2f}</span><br>"
            f"<span style='font-size:14px;white-space:pre-line;'>{w['text']}</span></div>",
            unsafe_allow_html=True,
        )


def render_llm_section(llm_result, me: str, partner: str):
    """analysis_result['llm']을 받아 AI 심층 분석 섹션을 그린다.

    llm_result 형태:
      None                                   → 키 없음/스킵
      {"error": str}                         → 호출 실패
      {"judgment","comparison","retrieved"}  → 정상
    """
    st.divider()
    st.subheader("🤖 AI 심층 분석 (LLM)")

    if llm_result is None:
        st.info("💡 사이드바에 OpenAI API 키를 입력하면 LLM 심층 분석이 활성화됩니다. (규칙 기반 결과는 위에 그대로 유지됩니다)")
        return

    if "error" in llm_result:
        st.warning(f"⚠️ LLM 분석을 완료하지 못했어요: {llm_result['error']} (규칙 기반 결과는 정상입니다)")
        return

    comparison = llm_result["comparison"]
    judgment = llm_result["judgment"]
    retrieved = llm_result["retrieved"]

    st.caption(f"LLM 신뢰도: {judgment.confidence:.0%}")
    _score_row("지배성", comparison["dominance"])
    _score_row("의존도", comparison["dependence"])

    with st.expander("🔍 LLM이 주목한 근거 대화"):
        _evidence_cards("지배성 근거", retrieved["dominance"])
        _evidence_cards("의존도 근거", retrieved["dependence"])

    st.markdown("### 📝 AI 진단 리포트")
    st.markdown(judgment.report)
```

- [ ] **Step 2: import 스모크 확인**

Run: `uv run python -c "import llm.ui; print('ok')"`
Expected: `ok` (구문/임포트 오류 없음)

- [ ] **Step 3: Commit**

```bash
git add llm/ui.py
git commit -m "feat(llm): add Streamlit AI deep-analysis section"
```

---

## Task 10: `app.py` 통합 — 사이드바 키 + Tier2 호출 + 렌더

**Files:**
- Modify: `app.py` (imports 상단, `render_loading`, `render_result`, 메인 진입부)

> 참고 (현재 코드 위치, 2026-06-05 기준):
> - import 블록: `app.py:1-17`
> - `render_loading`: `app.py:141-252` — `analysis_result` dict 생성은 `:226-244`, `phase="result"` 설정은 `:245`
> - `render_result`: `app.py:276` 시작
> - `sbert_model`은 `render_loading` 내 `:182`에서 로드됨

- [ ] **Step 1: import 추가**

`app.py` 상단 import 블록(기존 `from visualize.charts import (...)` 블록 다음 줄)에 추가:

```python
from llm.config import load_llm_config
from llm.client import LLMError
from llm.pipeline import run_llm_analysis
from llm.ui import render_llm_section
from models.hugging_face import encode_sentences
```

- [ ] **Step 2: 사이드바 API 키 입력 추가**

`app.py` 맨 아래의 phase 분기(모듈 레벨 또는 `main()` 내에서 `_init_state()` 후 `render_upload`/`render_loading`/`render_result`를 호출하는 부분)를 찾는다. 그 **phase 분기 직전**에 사이드바 블록을 추가:

```python
  with st.sidebar:
    st.markdown("### ⚙️ LLM 설정")
    api_key_input = st.text_input(
      "OpenAI API 키",
      type="password",
      help="입력하면 AI 심층 분석(Tier2)이 활성화됩니다. 키는 세션에만 보관되며 저장되지 않습니다.",
    )
    st.session_state["api_key"] = api_key_input or None
    st.caption("키 없이도 규칙 기반 분석은 정상 동작합니다.")
```

> 사이드바는 매 rerun마다 그려져야 하므로 phase 분기와 같은 스코프에서 분기 전에 둔다. 들여쓰기는 주변 코드(2칸)에 맞춘다.

- [ ] **Step 3: `render_loading`에 Tier2 단계 추가**

`render_loading` 함수에서 `st.session_state.analysis_result = { ... }` dict 대입(`:226-244`) 직후, `st.session_state.phase = "result"`(`:245`) **직전**에 삽입:

```python
    # Step 7: LLM 심층 분석 (Tier2) — 키 있을 때만, 실패해도 Tier1 보존
    llm_result = None
    api_key = st.session_state.get("api_key")
    if api_key:
      status.markdown("### 🤖 LLM 심층 분석 중...")
      try:
        config = load_llm_config(api_key_override=api_key)
        encoder = lambda texts: encode_sentences(texts, sbert_model)
        llm_result = run_llm_analysis(
          df_filtered, me, st.session_state.analysis_result, encoder, config
        )
      except LLMError as e:
        llm_result = {"error": str(e)}
      except Exception as e:  # 예기치 못한 오류도 Tier1은 보존
        llm_result = {"error": f"예상치 못한 오류: {e}"}

    st.session_state.analysis_result["llm"] = llm_result
```

> `sbert_model`은 같은 함수 `:182`에서 이미 로드되어 스코프에 있다. `st.session_state.analysis_result`는 바로 위에서 생성되어 `dominance_index`/`dependence_index`/`emotion_result`를 포함한다. 들여쓰기는 함수 본문(4칸)에 맞춘다.

- [ ] **Step 4: `render_result`에 섹션 렌더 추가**

`render_result` 함수 **맨 끝**(마지막 탭/콘텐츠 이후)에 추가:

```python
  # AI 심층 분석 섹션 (있으면)
  render_llm_section(r.get("llm"), me, partner)
```

> `r`은 `render_result` 시작부의 `r = st.session_state.analysis_result`, `me`/`partner`는 `r["me"]`/`r["partner"]`로 이미 정의됨. 들여쓰기 2칸.

- [ ] **Step 5: 구문 확인**

Run: `uv run python -c "import ast; ast.parse(open('app.py').read()); print('app.py syntax ok')"`
Expected: `app.py syntax ok`

- [ ] **Step 6: 앱 구동 스모크 (선택)**

Run:
```bash
uv run streamlit run app.py --server.headless true >/tmp/st.log 2>&1 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" localhost:8501
kill %1
```
Expected: `200` (앱이 기동되고 200 응답)

- [ ] **Step 7: 전체 테스트 회귀**

Run: `uv run pytest -q`
Expected: PASS (전체 통과)

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat(app): integrate Tier2 LLM analysis with sidebar key and graceful degradation"
```

---

## Task 11: README 업데이트 (포트폴리오 서술)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 2-tier 아키텍처 섹션 추가**

`README.md`의 "아키텍처" 섹션 뒤에 다음을 추가:

```markdown
## 🤖 2-Tier 하이브리드 분석 (LLM)

비용/스케일을 고려한 2계층 구조로 분석합니다.

| 계층 | 도구 | 역할 |
|------|------|------|
| **Tier 1** | KLUE-BERT · KR-SBERT · 규칙 | 전체 대화를 싸게 스캔해 지표를 계산 (베이스라인) |
| **RAG** | KR-SBERT 임베딩 | 축별 쿼리로 "의미 있는 근거 대화"만 검색 |
| **Tier 2** | OpenAI LLM | 검색된 근거로 점수 재판단 + 자연어 진단 리포트 |

- Tier1 점수와 LLM 점수를 **나란히 비교**하고, 불일치 시 LLM이 근거와 함께 이유를 설명합니다.
- LLM 인용은 검색 근거에 실제 존재하는지 **사후 검증**해 할루시네이션을 거릅니다.
- 전송 전 화자명을 `나`/`상대`로 **익명화**합니다.
- **OpenAI API 키 없이도 Tier1 분석은 정상 동작**합니다 (사이드바에서 키 입력 시 Tier2 활성화).

### 환경 변수
| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OPENAI_API_KEY` | — | LLM 호출용 키 (또는 사이드바 입력) |
| `LLM_MODEL` | `gpt-4o-mini` | 사용할 OpenAI 모델 |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document 2-tier LLM hybrid architecture"
```

---

## 완료 기준 (Definition of Done)

- [ ] `uv run pytest -q` 전체 통과 (config/schema/anonymize/retrieval/client/judge/compare/pipeline)
- [ ] 키 없이 앱 실행 시 Tier1 결과 정상 + "키 입력 안내" 표시
- [ ] 키 입력 시 비교 점수 2개 축 + 근거 카드 + 진단 리포트 표시
- [ ] 잘못된 키/네트워크 오류 시 앱이 죽지 않고 경고 + Tier1 보존
- [ ] (스트레치) LangGraph verify-refine 루프는 별도 작업으로 분리 — 본 계획 범위 밖

---

## 스트레치 골 (시간 남을 때, 본 계획 외)

`llm/graph.py`에 LangGraph로 `judge → critic(근거 검증) → 약하면 re-retrieve 후 재판단` 루프 구성. `langgraph` 의존성 추가 필요. 핵심 파이프라인(`run_llm_analysis`)을 대체하지 않고 옵션 경로로 둔다. 별도 spec/plan 사이클 권장.
