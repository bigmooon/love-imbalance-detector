# React + TypeScript + SCSS 프론트엔드 전환 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit 앱을 FastAPI 백엔드 + React/TS/SCSS 프론트엔드로 전환하고, "에디토리얼 리포트" 디자인(세리프 디스플레이 + 모노 데이터, 크림/잉크 + 시그널 레드)으로 전체 플로우(업로드 → 분석 → 결과 리포트)를 구현한다.

**Architecture:** 기존 Python 분석 모듈(features/, models/, llm/)은 그대로 두고, `server/` 패키지가 FastAPI로 감싼다. 분석은 백그라운드 잡으로 실행되고 프론트가 진행 상태를 폴링한다. 차트는 Plotly figure 대신 **JSON 데이터 시리즈**를 API로 보내고 React(Recharts + 커스텀 SVG/CSS)가 렌더링한다. Streamlit 앱(app.py)은 레거시 데모로 유지한다.

**Tech Stack:** FastAPI + Uvicorn (백엔드), Vite + React 18 + TypeScript + SCSS Modules + Recharts + react-markdown (프론트), pytest + Vitest (테스트)

**디자인 방향 (에디토리얼 리포트):**
- 컨셉: "관계 감사 보고서(RELATIONSHIP AUDIT)" — 잡지/진단 리포트 문서 느낌
- 타이포: 디스플레이 = Noto Serif KR, 데이터/라벨 = IBM Plex Mono, 본문 = Pretendard
- 컬러: 크림 종이 `#F6F1E7` / 잉크 `#1C1814` / 시그널 레드 `#C8361F` / 나(me) = 잉크 블루 `#1F4D8F`, 상대(partner) = 시그널 레드
- 레이아웃: 번호 매긴 섹션(01—05), 헤어라인 룰, 큰 세리프 헤드라인, 모노 데이터 테이블
- 모션: 리포트 로드 시 섹션별 스태거 페이드 인(한 번의 잘 연출된 로드 시퀀스), 분석 중 화면의 세리프 카운터

---

## API 계약 (전체 작업의 기준 인터페이스)

```
POST /api/upload          multipart(file: csv)        → UploadSummary
POST /api/analyze         AnalyzeRequest(JSON)        → {"job_id": str}
GET  /api/jobs/{job_id}                               → JobStatus (result 포함 가능)
```

`ReportPayload` 핵심 구조 (Task 2에서 Pydantic으로, Task 8에서 TS 타입으로 정의):

```jsonc
{
  "me": "지언", "partner": "민수",
  "dominance_index": 0.62, "dependence_index": 0.71, "balance": 0.91,
  "radar": {"categories": ["선톡 비율", ...7개], "me": [0.6, ...], "partner": [0.4, ...]},
  "participation": {"message_count_ratio": 0.55, "char_count_ratio": 0.6, "avg_length_me": 18.2, "avg_length_partner": 12.1},
  "timeline": [{"week": "2025-01-06", "me": 12, "partner": 18}, ...],
  "emotion": {"me": {"joy": 0.4, "anger": 0.1, ...6키}, "partner": {...}, "joy_gap": 0.1, "negative_gap": -0.05},
  "reply_time": {
    "me_median_sec": 35.0, "partner_median_sec": 180.0,
    "me_box": {"lo": 0.2, "q1": 0.5, "median": 0.6, "q3": 2.0, "hi": 30.0},   // 분 단위
    "partner_box": {...}
  },
  "double_text": {"me": 0.21, "partner": 0.13},
  "initiation_ratio": 0.7,
  "qa_sincerity": {"avg_sincerity": 0.55, "my_sincerity": 0.6, "partner_sincerity": 0.5,
                   "pairs": [{"questioner": "지언", "question": "...", "answerer": "민수", "answer": "...", "score": 0.3}]},
  "llm": null,            // 또는 LLMPayload
  "llm_error": null       // LLM 호출 실패 시 에러 메시지 (Tier1 결과는 보존)
}
```

`LLMPayload`:

```jsonc
{
  "confidence": 0.8,
  "report": "## 마크다운 리포트...",
  "dominance": {"tier1": 0.62, "llm": 0.7, "delta": 0.08, "agree": true},
  "dependence": {"tier1": 0.71, "llm": 0.5, "delta": -0.21, "agree": false},
  "evidence": {"dominance": [{"text": "[나] ...", "sim": 0.72}], "dependence": [...]}
}
```

`JobStatus`:

```jsonc
{"job_id": "...", "status": "pending|running|done|error",
 "step": 3, "total_steps": 7, "label": "감정 분류 중",
 "result": null, "error": null}
```

---

### Task 0: 작업 브랜치 생성

**Files:** 없음 (git만)

- [ ] **Step 1: 현재 브랜치에서 새 브랜치 생성**

Run: `git checkout -b feat/react-frontend`
Expected: `Switched to a new branch 'feat/react-frontend'` (feat/2tier-llm-hybrid 기반 — LLM 기능에 의존하므로)

---

### Task 1: 분석 모듈에서 Streamlit 결합 제거

`models/hugging_face.py`와 `utils/kakao_parser.py`가 `streamlit`을 import하고 있어 FastAPI 프로세스에서 쓸 수 없다(ScriptRunContext 경고/불필요 의존). `functools.lru_cache`로 대체한다. 또한 `WEIGHT_PRESETS`를 app.py에서 공용 모듈로 옮긴다.

**Files:**
- Modify: `models/hugging_face.py:1-32` (streamlit 제거, lru_cache로 교체)
- Modify: `utils/kakao_parser.py:1-10` (streamlit 제거)
- Create: `features/presets.py`
- Modify: `app.py:2,29-43` (WEIGHT_PRESETS import로 교체)
- Test: `tests/test_decoupling.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_decoupling.py
"""분석 모듈이 Streamlit 없이 import 가능한지 검증."""
import subprocess
import sys


def test_models_importable_without_streamlit():
    """models.hugging_face가 streamlit을 import하지 않아야 한다."""
    code = (
        "import sys; import models.hugging_face; "
        "assert 'streamlit' not in sys.modules, 'streamlit imported!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parser_importable_without_streamlit():
    code = (
        "import sys; import utils.kakao_parser; "
        "assert 'streamlit' not in sys.modules, 'streamlit imported!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_weight_presets_moved_to_features():
    from features.presets import WEIGHT_PRESETS
    assert "기본" in WEIGHT_PRESETS
    assert WEIGHT_PRESETS["기본"] == {"dominance": None, "dependence": None}
    assert "답장속도 중시" in WEIGHT_PRESETS
    assert "감정 중시" in WEIGHT_PRESETS
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_decoupling.py -v`
Expected: FAIL — `streamlit imported!` 2건 + `ModuleNotFoundError: features.presets`

- [ ] **Step 3: models/hugging_face.py 수정**

파일 상단(1~32행)을 다음으로 교체. `import streamlit as st` 제거, `@st.cache_resource` → `@lru_cache`:

```python
from functools import lru_cache

import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from models.emotion_labels import LABEL2ID, get_emotion_group, ACTIVE_PRESET

BATCH_SIZE = 64
SBERT_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"


@lru_cache(maxsize=1)
def load_emotion_classifier():
  """
  HuggingFace 감정 분류 파이프라인 로드.
  @lru_cache로 프로세스 내 1회만 로드 (Streamlit/FastAPI 공용).
  """
  return pipeline(
    "text-classification",
    model=ACTIVE_PRESET,
    top_k=1,
    truncation=True,
    max_length=512,
  )


@lru_cache(maxsize=1)
def load_sbert_model():
  """
  KR-SBERT 문장 임베딩 모델 로드.
  @lru_cache로 프로세스 내 1회만 로드 (Streamlit/FastAPI 공용).
  """
  return SentenceTransformer(SBERT_MODEL_NAME)
```

이하 `classify_emotions`, `encode_sentences`는 그대로 둔다.

- [ ] **Step 4: utils/kakao_parser.py 수정**

1~9행에서 `import streamlit as st`와 `@st.cache_data(show_spinner=False)` 데코레이터 제거:

```python
import pandas as pd
from utils.text_utils import is_system_message, is_non_text, clean_text

SESSION_END_MINUTES = 30
MAX_USERS = 2
BOT = ["플레이봇"]

def parse_kakao_chat(file_path):
```

(함수 본문은 변경 없음. 캐싱은 Streamlit 전용 최적화였고, 업로드는 1회성이므로 제거해도 무방.)

- [ ] **Step 5: features/presets.py 생성**

app.py 29~43행의 `WEIGHT_PRESETS`를 그대로 이동:

```python
# features/presets.py
"""지배성/의존도 가중치 프리셋. None이면 각 compute 함수의 DEFAULT_WEIGHTS 사용."""

WEIGHT_PRESETS = {
  "기본": {"dominance": None, "dependence": None},
  "답장속도 중시": {
    "dominance": None,
    "dependence": {"reply_time_ratio": 0.55, "double_text_ratio": 0.25, "qa_sincerity_gap": 0.20},
  },
  "감정 중시": {
    "dominance": {
      "initiation_ratio": 0.10, "ending_ratio": 0.10,
      "message_count_ratio": 0.10, "char_count_ratio": 0.05,
      "joy_gap": 0.30, "negative_gap": 0.35,
    },
    "dependence": None,
  },
}
```

- [ ] **Step 6: app.py에서 프리셋 import로 교체**

app.py의 `WEIGHT_PRESETS = {...}` 블록(29~43행)을 삭제하고 import 추가:

```python
from features.presets import WEIGHT_PRESETS
```

- [ ] **Step 7: 전체 테스트 통과 확인**

Run: `python -m pytest tests/ -v`
Expected: 기존 llm 테스트 + 신규 test_decoupling.py 모두 PASS

- [ ] **Step 8: 커밋**

```bash
git add models/hugging_face.py utils/kakao_parser.py features/presets.py app.py tests/test_decoupling.py
git commit -m "refactor: decouple analysis modules from streamlit for API reuse"
```

---

### Task 2: server/schemas.py — API 응답/요청 Pydantic 모델

**Files:**
- Create: `server/__init__.py` (빈 파일)
- Create: `server/schemas.py`
- Test: `tests/server/__init__.py` (빈 파일), `tests/server/conftest.py`, `tests/server/test_schemas.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/server/test_schemas.py
import pytest
from pydantic import ValidationError

from server.schemas import AnalyzeRequest, ReportPayload, JobStatus


def test_analyze_request_defaults():
    req = AnalyzeRequest(
        upload_id="u1", me="지언",
        start_date="2025-01-01", end_date="2025-06-01",
    )
    assert req.session_gap == 30
    assert req.preset == "기본"
    assert req.api_key is None


def test_analyze_request_rejects_bad_session_gap():
    with pytest.raises(ValidationError):
        AnalyzeRequest(
            upload_id="u1", me="지언",
            start_date="2025-01-01", end_date="2025-06-01",
            session_gap=5,  # 최소 10
        )


def test_job_status_literal():
    js = JobStatus(job_id="j1", status="running", step=2, total_steps=7, label="모델 로딩 중")
    assert js.result is None
    with pytest.raises(ValidationError):
        JobStatus(job_id="j1", status="banana", step=0, total_steps=7, label="x")


def test_report_payload_roundtrip(sample_payload_dict):
    payload = ReportPayload.model_validate(sample_payload_dict)
    assert payload.me == "지언"
    assert len(payload.radar.categories) == 7
    assert payload.llm is None
```

그리고 공용 픽스처:

```python
# tests/server/conftest.py
import pytest


@pytest.fixture
def sample_payload_dict():
    box = {"lo": 0.2, "q1": 0.5, "median": 1.0, "q3": 2.0, "hi": 10.0}
    return {
        "me": "지언", "partner": "민수",
        "dominance_index": 0.62, "dependence_index": 0.71, "balance": 0.91,
        "radar": {
            "categories": ["선톡 비율", "대화 종료", "메시지 비율", "글자 비율", "답장 속도", "더블텍스트", "QA 성의도"],
            "me": [0.6, 0.5, 0.55, 0.6, 0.7, 0.4, 0.5],
            "partner": [0.4, 0.5, 0.45, 0.4, 0.3, 0.6, 0.5],
        },
        "participation": {
            "message_count_ratio": 0.55, "char_count_ratio": 0.6,
            "avg_length_me": 18.2, "avg_length_partner": 12.1,
        },
        "timeline": [{"week": "2025-01-06", "me": 12, "partner": 18}],
        "emotion": {
            "me": {"joy": 0.4, "anger": 0.1, "sadness": 0.1, "anxiety": 0.2, "hurt": 0.1, "embarrass": 0.1},
            "partner": {"joy": 0.5, "anger": 0.1, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "joy_gap": -0.1, "negative_gap": 0.1,
        },
        "reply_time": {
            "me_median_sec": 35.0, "partner_median_sec": 180.0,
            "me_box": box, "partner_box": box,
        },
        "double_text": {"me": 0.21, "partner": 0.13},
        "initiation_ratio": 0.7,
        "qa_sincerity": {
            "avg_sincerity": 0.55, "my_sincerity": 0.6, "partner_sincerity": 0.5,
            "pairs": [{"questioner": "민수", "question": "뭐해?", "answerer": "지언", "answer": "일해", "score": 0.3}],
        },
        "llm": None,
        "llm_error": None,
    }
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/server/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: server/schemas.py 구현**

```python
# server/schemas.py
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class UploadSummary(BaseModel):
    upload_id: str
    users: list[str]
    message_count: int
    first_date: date
    last_date: date


class AnalyzeRequest(BaseModel):
    upload_id: str
    me: str
    start_date: date
    end_date: date
    session_gap: int = Field(default=30, ge=10, le=120)
    preset: str = "기본"
    api_key: str | None = None


class RadarPayload(BaseModel):
    categories: list[str]
    me: list[float]
    partner: list[float]


class Participation(BaseModel):
    message_count_ratio: float
    char_count_ratio: float
    avg_length_me: float
    avg_length_partner: float


class TimelinePoint(BaseModel):
    week: str  # ISO 날짜 (주 시작일)
    me: int
    partner: int


class EmotionPayload(BaseModel):
    me: dict[str, float]
    partner: dict[str, float]
    joy_gap: float
    negative_gap: float


class BoxStats(BaseModel):
    """답장 시간 분포 (분 단위). lo/hi는 5/95 퍼센타일."""
    lo: float
    q1: float
    median: float
    q3: float
    hi: float


class ReplyTimePayload(BaseModel):
    me_median_sec: float       # partner가 말한 뒤 me가 답하기까지 중앙값
    partner_median_sec: float  # me가 말한 뒤 partner가 답하기까지 중앙값
    me_box: BoxStats
    partner_box: BoxStats


class PairRatio(BaseModel):
    me: float
    partner: float


class QAPair(BaseModel):
    questioner: str
    question: str
    answerer: str
    answer: str
    score: float


class QASincerityPayload(BaseModel):
    avg_sincerity: float
    my_sincerity: float
    partner_sincerity: float
    pairs: list[QAPair]


class AxisComparison(BaseModel):
    tier1: float
    llm: float
    delta: float
    agree: bool


class EvidenceWindow(BaseModel):
    text: str
    sim: float


class LLMPayload(BaseModel):
    confidence: float
    report: str
    dominance: AxisComparison
    dependence: AxisComparison
    evidence: dict[str, list[EvidenceWindow]]


class ReportPayload(BaseModel):
    me: str
    partner: str
    dominance_index: float
    dependence_index: float
    balance: float
    radar: RadarPayload
    participation: Participation
    timeline: list[TimelinePoint]
    emotion: EmotionPayload
    reply_time: ReplyTimePayload
    double_text: PairRatio
    initiation_ratio: float
    qa_sincerity: QASincerityPayload
    llm: LLMPayload | None = None
    llm_error: str | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    step: int
    total_steps: int
    label: str
    result: ReportPayload | None = None
    error: str | None = None
```

`server/__init__.py`, `tests/server/__init__.py`는 빈 파일로 생성.

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/server/ -v`
Expected: 4 PASS

- [ ] **Step 5: 커밋**

```bash
git add server/ tests/server/
git commit -m "feat(server): add API request/response schemas"
```

---

### Task 3: server/serialize.py — 분석 결과 dict → ReportPayload 변환

app.py의 `analysis_result` dict(차트 figure 제외)와 `df_filtered`로부터 JSON 직렬화 가능한 `ReportPayload`를 만든다. 주간 타임라인·답장 박스 통계를 여기서 계산한다.

**Files:**
- Create: `server/serialize.py`
- Test: `tests/server/test_serialize.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/server/test_serialize.py
import pandas as pd
import pytest

from server.serialize import build_report_payload


@pytest.fixture
def df_filtered():
    rows = [
        ("2025-01-06 10:00:00", "지언", "안녕 뭐해?", 0),
        ("2025-01-06 10:00:30", "민수", "일하지", 0),
        ("2025-01-06 10:05:00", "지언", "점심 먹었어?", 0),
        ("2025-01-06 10:20:00", "민수", "응", 0),
        ("2025-01-13 09:00:00", "지언", "주말에 뭐했어?", 1),
        ("2025-01-13 09:30:00", "민수", "그냥 쉬었어", 1),
    ]
    df = pd.DataFrame(rows, columns=["Date", "User", "Message", "Session_ID"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def analysis_result(df_filtered):
    return {
        "me": "지언", "partner": "민수", "df_filtered": df_filtered,
        "dominance_metrics": {
            "initiation_ratio": 1.0, "ending_ratio": 0.0,
            "message_count_ratio": 0.5, "char_count_ratio": 0.6,
            "joy_gap": 0.1, "negative_gap": -0.05,
        },
        "dependence_metrics": {
            "reply_time_ratio": 2.0, "double_text_ratio": 0.0, "qa_sincerity_gap": 0.1,
        },
        "dominance_index": 0.62, "dependence_index": 0.71,
        "emotion_result": {
            "me": {"joy": 0.5, "anger": 0.1, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "partner": {"joy": 0.4, "anger": 0.2, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "joy_gap": 0.1, "negative_gap": -0.05,
        },
        "reply_time": {"me_to_partner_median_sec": 465.0, "partner_to_me_median_sec": 30.0, "ratio": 0.06},
        "double_text": 0.33, "double_text_partner": 0.0,
        "qa_sincerity": {
            "gap": 0.1, "my_sincerity": 0.6, "partner_sincerity": 0.5, "avg_sincerity": 0.55,
            "all_pairs": [
                {"questioner": "지언", "question": "뭐해?", "answerer": "민수", "answer": "일하지", "score": 0.3},
            ] * 15,  # 15개 → 10개로 잘리는지 확인
        },
        "participation": {
            "message_count_ratio": 0.5, "char_count_ratio": 0.6,
            "avg_length_me": 8.0, "avg_length_partner": 4.0,
        },
        "llm": None,
    }


def test_basic_fields(analysis_result):
    payload = build_report_payload(analysis_result)
    assert payload.me == "지언"
    assert payload.partner == "민수"
    assert payload.balance == pytest.approx(1 - abs(0.62 - 0.71))


def test_radar_has_seven_axes(analysis_result):
    payload = build_report_payload(analysis_result)
    assert len(payload.radar.categories) == 7
    assert len(payload.radar.me) == 7
    # partner = 1 - me (app.py 레이더와 동일 규칙)
    assert payload.radar.partner[0] == pytest.approx(1 - payload.radar.me[0])


def test_timeline_weekly_counts(analysis_result):
    payload = build_report_payload(analysis_result)
    weeks = {p.week: p for p in payload.timeline}
    assert "2025-01-06" in weeks
    assert weeks["2025-01-06"].me == 2
    assert weeks["2025-01-06"].partner == 2
    assert weeks["2025-01-13"].me == 1


def test_reply_time_mapping(analysis_result):
    payload = build_report_payload(analysis_result)
    # me_median = partner_to_me (내가 답장하기까지)
    assert payload.reply_time.me_median_sec == 30.0
    assert payload.reply_time.partner_median_sec == 465.0
    assert payload.reply_time.me_box.median > 0


def test_qa_pairs_capped_at_ten(analysis_result):
    payload = build_report_payload(analysis_result)
    assert len(payload.qa_sincerity.pairs) == 10


def test_llm_none_and_error(analysis_result):
    assert build_report_payload(analysis_result).llm is None
    analysis_result["llm"] = {"error": "boom"}
    payload = build_report_payload(analysis_result)
    assert payload.llm is None
    assert payload.llm_error == "boom"


def test_llm_full(analysis_result):
    from llm.schema import LLMJudgment, AxisJudgment
    judgment = LLMJudgment(
        dominance=AxisJudgment(score=0.7, rationale="r", evidence=[]),
        dependence=AxisJudgment(score=0.5, rationale="r", evidence=[]),
        report="## 리포트", confidence=0.8,
    )
    analysis_result["llm"] = {
        "judgment": judgment,
        "comparison": {
            "dominance": {"tier1": 0.62, "llm": 0.7, "delta": 0.08, "agree": True},
            "dependence": {"tier1": 0.71, "llm": 0.5, "delta": -0.21, "agree": False},
            "agreement_threshold": 0.15,
        },
        "retrieved": {
            "dominance": [{"text": "[나] 보고싶어", "speakers": ["나"], "session_id": 0, "sim": 0.7}],
            "dependence": [],
        },
    }
    payload = build_report_payload(analysis_result)
    assert payload.llm.confidence == 0.8
    assert payload.llm.dominance.agree is True
    assert payload.llm.evidence["dominance"][0].text == "[나] 보고싶어"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/server/test_serialize.py -v`
Expected: FAIL — `ModuleNotFoundError: server.serialize`

- [ ] **Step 3: server/serialize.py 구현**

```python
# server/serialize.py
"""분석 결과 dict(app.py의 analysis_result 형태)를 ReportPayload로 변환."""
import numpy as np
import pandas as pd

from visualize.charts import RADAR_CATEGORIES, _normalize_for_radar
from server.schemas import (
    AxisComparison, BoxStats, EmotionPayload, EvidenceWindow, LLMPayload,
    PairRatio, Participation, QAPair, QASincerityPayload, RadarPayload,
    ReplyTimePayload, ReportPayload, TimelinePoint,
)

MAX_QA_PAIRS = 10


def _build_radar(dominance_metrics: dict, dependence_metrics: dict) -> RadarPayload:
    me_values = _normalize_for_radar(dominance_metrics, dependence_metrics)
    return RadarPayload(
        categories=list(RADAR_CATEGORIES),
        me=[float(v) for v in me_values],
        partner=[float(1 - v) for v in me_values],
    )


def _build_timeline(df: pd.DataFrame, me: str, partner: str) -> list[TimelinePoint]:
    df_week = df.copy()
    df_week["Week"] = df_week["Date"].dt.to_period("W").dt.start_time
    counts = df_week.groupby(["Week", "User"]).size().unstack(fill_value=0)
    return [
        TimelinePoint(
            week=week.date().isoformat(),
            me=int(row.get(me, 0)),
            partner=int(row.get(partner, 0)),
        )
        for week, row in counts.iterrows()
    ]


def _reply_minutes(df: pd.DataFrame, replier: str, original: str) -> pd.Series:
    """original이 말한 뒤 같은 세션에서 replier가 답하기까지 걸린 시간(분)."""
    prev_user = df["User"].shift(1)
    prev_date = df["Date"].shift(1)
    prev_session = df["Session_ID"].shift(1)
    is_reply = (
        (df["User"] == replier) & (prev_user == original)
        & (df["Session_ID"] == prev_session)
    )
    return (df["Date"] - prev_date).dt.total_seconds()[is_reply] / 60


def _box_stats(minutes: pd.Series) -> BoxStats:
    if len(minutes) == 0:
        return BoxStats(lo=0.0, q1=0.0, median=0.0, q3=0.0, hi=0.0)
    lo, q1, med, q3, hi = np.percentile(minutes, [5, 25, 50, 75, 95])
    return BoxStats(lo=float(lo), q1=float(q1), median=float(med), q3=float(q3), hi=float(hi))


def _build_reply_time(df: pd.DataFrame, me: str, partner: str, reply_time: dict) -> ReplyTimePayload:
    return ReplyTimePayload(
        me_median_sec=float(reply_time["partner_to_me_median_sec"]),
        partner_median_sec=float(reply_time["me_to_partner_median_sec"]),
        me_box=_box_stats(_reply_minutes(df, me, partner)),
        partner_box=_box_stats(_reply_minutes(df, partner, me)),
    )


def _build_llm(llm_result) -> tuple[LLMPayload | None, str | None]:
    if llm_result is None:
        return None, None
    if "error" in llm_result:
        return None, str(llm_result["error"])
    judgment = llm_result["judgment"]
    comparison = llm_result["comparison"]
    retrieved = llm_result["retrieved"]
    return LLMPayload(
        confidence=float(judgment.confidence),
        report=judgment.report,
        dominance=AxisComparison(**comparison["dominance"]),
        dependence=AxisComparison(**comparison["dependence"]),
        evidence={
            axis: [EvidenceWindow(text=w["text"], sim=float(w["sim"])) for w in windows[:4]]
            for axis, windows in retrieved.items()
        },
    ), None


def build_report_payload(result: dict) -> ReportPayload:
    me, partner = result["me"], result["partner"]
    df = result["df_filtered"]
    dom, dep = float(result["dominance_index"]), float(result["dependence_index"])
    qa = result["qa_sincerity"]
    llm_payload, llm_error = _build_llm(result.get("llm"))

    return ReportPayload(
        me=me,
        partner=partner,
        dominance_index=dom,
        dependence_index=dep,
        balance=float(1 - abs(dom - dep)),
        radar=_build_radar(result["dominance_metrics"], result["dependence_metrics"]),
        participation=Participation(**result["participation"]),
        timeline=_build_timeline(df, me, partner),
        emotion=EmotionPayload(**result["emotion_result"]),
        reply_time=_build_reply_time(df, me, partner, result["reply_time"]),
        double_text=PairRatio(me=float(result["double_text"]), partner=float(result["double_text_partner"])),
        initiation_ratio=float(result["dominance_metrics"]["initiation_ratio"]),
        qa_sincerity=QASincerityPayload(
            avg_sincerity=float(qa["avg_sincerity"]),
            my_sincerity=float(qa["my_sincerity"]),
            partner_sincerity=float(qa["partner_sincerity"]),
            pairs=[QAPair(**p) for p in qa["all_pairs"][:MAX_QA_PAIRS]],
        ),
        llm=llm_payload,
        llm_error=llm_error,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/server/test_serialize.py -v`
Expected: 7 PASS

- [ ] **Step 5: 커밋**

```bash
git add server/serialize.py tests/server/test_serialize.py
git commit -m "feat(server): serialize analysis result into JSON report payload"
```

---

### Task 4: server/analysis.py — 분석 파이프라인 추출

app.py `render_loading()`의 분석 로직을 UI 없는 함수로 추출한다. 진행 콜백과 모델 주입(테스트용)을 지원한다.

**Files:**
- Create: `server/analysis.py`
- Test: `tests/server/test_analysis.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/server/test_analysis.py
from datetime import date

import numpy as np
import pandas as pd
import pytest

from server.analysis import AnalysisOptions, run_analysis, PROGRESS_LABELS


@pytest.fixture
def raw_df():
    rows = []
    # 2개 세션, 질문-답변 쌍 충분히 포함
    base = pd.Timestamp("2025-01-06 10:00:00")
    msgs = [
        ("지언", "안녕 뭐해?"), ("민수", "일하지"), ("지언", "점심은 먹었어?"), ("민수", "응 먹었어"),
        ("지언", "오늘 저녁에 시간 돼?"), ("민수", "글쎄"), ("지언", "보고싶다"), ("민수", "나도"),
    ]
    for i, (user, msg) in enumerate(msgs):
        rows.append((base + pd.Timedelta(minutes=i), user, msg))
    df = pd.DataFrame(rows, columns=["Date", "User", "Message"])
    return df


def fake_classifier(batch):
    """기쁨 라벨 고정 반환 (HF pipeline 모사: 리스트의 리스트)."""
    return [[{"label": "기쁨", "score": 0.9}] for _ in batch]


class FakeSbert:
    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(42)
        return rng.random((len(texts), 8))


def test_run_analysis_returns_full_result(raw_df):
    opts = AnalysisOptions(
        me="지언", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
    )
    steps = []
    result = run_analysis(
        raw_df, opts,
        progress_cb=lambda i, label: steps.append((i, label)),
        classifier=fake_classifier, sbert_model=FakeSbert(),
    )
    assert result["me"] == "지언"
    assert result["partner"] == "민수"
    assert 0.0 <= result["dominance_index"] <= 1.0
    assert 0.0 <= result["dependence_index"] <= 1.0
    assert result["llm"] is None  # api_key 없음
    assert "df_filtered" in result
    # 진행 콜백이 순서대로 호출됨
    assert [i for i, _ in steps] == sorted([i for i, _ in steps])
    assert steps[0][1] == PROGRESS_LABELS[0]


def test_run_analysis_empty_range_raises(raw_df):
    opts = AnalysisOptions(
        me="지언", start_date=date(2030, 1, 1), end_date=date(2030, 12, 31),
    )
    with pytest.raises(ValueError, match="기간"):
        run_analysis(raw_df, opts, classifier=fake_classifier, sbert_model=FakeSbert())
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/server/test_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: server.analysis`

- [ ] **Step 3: server/analysis.py 구현**

```python
# server/analysis.py
"""UI 없는 분석 파이프라인. app.py render_loading()의 로직을 추출한 것."""
from dataclasses import dataclass
from datetime import date

from utils.kakao_parser import split_sessions
from features.presets import WEIGHT_PRESETS
from features.dominance import (
    calc_start_ratio, calc_end_ratio, calc_participation_ratio,
    calc_emotion_dominance, compute_dominance_features,
)
from features.dependence import (
    calc_reply_time_asymmetry, calc_double_text_ratio,
    calc_qa_sincerity, compute_dependence_index,
)
from llm.config import load_llm_config
from llm.client import LLMError
from llm.pipeline import run_llm_analysis
from models.hugging_face import encode_sentences

PROGRESS_LABELS = [
    "데이터 준비 중",
    "모델 로딩 중",
    "감정 분류 중",
    "임베딩 계산 중",
    "지표 계산 중",
    "리포트 구성 중",
    "LLM 심층 분석 중",
]
TOTAL_STEPS = len(PROGRESS_LABELS)


@dataclass(frozen=True)
class AnalysisOptions:
    me: str
    start_date: date
    end_date: date
    session_gap: int = 30
    preset: str = "기본"
    api_key: str | None = None


def run_analysis(df, opts: AnalysisOptions, progress_cb=lambda i, label: None,
                 classifier=None, sbert_model=None) -> dict:
    """전체 분석 실행. 반환 dict는 app.py analysis_result와 동일 구조(figure 제외)."""

    def step(i):
        progress_cb(i, PROGRESS_LABELS[i])

    # Step 0: 데이터 준비
    step(0)
    df_filtered = df[
        (df["Date"].dt.date >= opts.start_date)
        & (df["Date"].dt.date <= opts.end_date)
    ].copy().reset_index(drop=True)
    if df_filtered.empty:
        raise ValueError("선택한 기간에 메시지가 없습니다. 기간을 다시 확인해주세요.")

    df_filtered = split_sessions(df_filtered, opts.session_gap)
    others = [u for u in df_filtered["User"].unique() if u != opts.me]
    if not others:
        raise ValueError("본인 외 대화 참여자가 없습니다.")
    partner = others[0]

    # Step 1: 모델 로딩 (미주입 시 실제 모델 — lazy import로 테스트 부담 제거)
    step(1)
    if classifier is None or sbert_model is None:
        from models.hugging_face import load_emotion_classifier, load_sbert_model
        classifier = classifier or load_emotion_classifier()
        sbert_model = sbert_model or load_sbert_model()

    # Step 2: 감정 분류
    step(2)
    emotion_result = calc_emotion_dominance(df_filtered, opts.me, classifier)

    # Step 3: 임베딩 (QA 성의도)
    step(3)
    qa_result = calc_qa_sincerity(df_filtered, opts.me, sbert_model)

    # Step 4: 지표 계산
    step(4)
    participation = calc_participation_ratio(df_filtered, opts.me)
    reply_time = calc_reply_time_asymmetry(df_filtered, opts.me)
    dominance_metrics = {
        "initiation_ratio": calc_start_ratio(df_filtered, opts.me),
        "ending_ratio": calc_end_ratio(df_filtered, opts.me),
        "message_count_ratio": participation["message_count_ratio"],
        "char_count_ratio": participation["char_count_ratio"],
        "joy_gap": emotion_result["joy_gap"],
        "negative_gap": emotion_result["negative_gap"],
    }
    dependence_metrics = {
        "reply_time_ratio": reply_time["ratio"],
        "double_text_ratio": calc_double_text_ratio(df_filtered, opts.me),
        "qa_sincerity_gap": qa_result["gap"],
    }
    weights = WEIGHT_PRESETS.get(opts.preset, WEIGHT_PRESETS["기본"])

    # Step 5: 결과 조립
    step(5)
    result = {
        "me": opts.me,
        "partner": partner,
        "df_filtered": df_filtered,
        "dominance_metrics": dominance_metrics,
        "dependence_metrics": dependence_metrics,
        "dominance_index": compute_dominance_features(dominance_metrics, weights["dominance"]),
        "dependence_index": compute_dependence_index(dependence_metrics, weights["dependence"]),
        "emotion_result": emotion_result,
        "reply_time": reply_time,
        "double_text": dependence_metrics["double_text_ratio"],
        "double_text_partner": calc_double_text_ratio(df_filtered, partner),
        "qa_sincerity": qa_result,
        "participation": participation,
    }

    # Step 6: LLM 심층 분석 (키 있을 때만, 실패해도 Tier1 보존 — app.py와 동일 정책)
    llm_result = None
    if opts.api_key:
        step(6)
        try:
            config = load_llm_config(api_key_override=opts.api_key)
            encoder = lambda texts: encode_sentences(texts, sbert_model)
            llm_result = run_llm_analysis(df_filtered, opts.me, result, encoder, config)
        except LLMError as e:
            llm_result = {"error": str(e)}
        except Exception as e:  # 예기치 못한 오류도 Tier1은 보존
            llm_result = {"error": f"예상치 못한 오류: {e}"}

    return {**result, "llm": llm_result}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/server/test_analysis.py -v`
Expected: 2 PASS

- [ ] **Step 5: 커밋**

```bash
git add server/analysis.py tests/server/test_analysis.py
git commit -m "feat(server): extract UI-free analysis pipeline with progress callback"
```

---

### Task 5: server/store.py — 업로드/잡 인메모리 저장소

**Files:**
- Create: `server/store.py`
- Test: `tests/server/test_store.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/server/test_store.py
import pandas as pd

from server.store import UploadStore, JobStore


def test_upload_store_put_get():
    store = UploadStore(max_items=2)
    df = pd.DataFrame({"a": [1]})
    uid = store.put(df)
    assert store.get(uid) is df


def test_upload_store_evicts_oldest():
    store = UploadStore(max_items=2)
    ids = [store.put(pd.DataFrame({"a": [i]})) for i in range(3)]
    assert store.get(ids[0]) is None  # 가장 오래된 것 축출
    assert store.get(ids[2]) is not None


def test_upload_store_missing_returns_none():
    assert UploadStore().get("nope") is None


def test_job_lifecycle():
    store = JobStore()
    job_id = store.create(total_steps=7)
    job = store.get(job_id)
    assert job.status == "pending"

    store.set_progress(job_id, step=2, label="감정 분류 중")
    job = store.get(job_id)
    assert job.status == "running"
    assert job.step == 2
    assert job.label == "감정 분류 중"

    store.set_done(job_id, result={"ok": True})
    job = store.get(job_id)
    assert job.status == "done"
    assert job.result == {"ok": True}


def test_job_error():
    store = JobStore()
    job_id = store.create(total_steps=7)
    store.set_error(job_id, "boom")
    job = store.get(job_id)
    assert job.status == "error"
    assert job.error == "boom"


def test_job_missing_returns_none():
    assert JobStore().get("nope") is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/server/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: server.store`

- [ ] **Step 3: server/store.py 구현**

레코드는 불변 dataclass로 두고 갱신 시 `dataclasses.replace`로 새 객체 생성(변경 금지 원칙):

```python
# server/store.py
"""업로드된 DataFrame과 분석 잡의 인메모리 저장소 (단일 프로세스용)."""
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any


class UploadStore:
    """업로드 CSV의 파싱 결과 DataFrame 보관. max_items 초과 시 가장 오래된 것 축출."""

    def __init__(self, max_items: int = 20):
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._max_items = max_items
        self._lock = threading.Lock()

    def put(self, df) -> str:
        upload_id = uuid.uuid4().hex
        with self._lock:
            self._items[upload_id] = df
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return upload_id

    def get(self, upload_id: str):
        with self._lock:
            return self._items.get(upload_id)


@dataclass(frozen=True)
class Job:
    job_id: str
    total_steps: int
    status: str = "pending"
    step: int = 0
    label: str = "대기 중"
    result: Any = None
    error: str | None = None


class JobStore:
    """분석 잡 상태 보관. 갱신은 항상 새 Job 객체로 교체(불변)."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, total_steps: int) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id, total_steps=total_steps)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, **changes)

    def set_progress(self, job_id: str, step: int, label: str):
        self._update(job_id, status="running", step=step, label=label)

    def set_done(self, job_id: str, result):
        self._update(job_id, status="done", result=result, label="완료")

    def set_error(self, job_id: str, error: str):
        self._update(job_id, status="error", error=error, label="오류")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/server/test_store.py -v`
Expected: 6 PASS

- [ ] **Step 5: 커밋**

```bash
git add server/store.py tests/server/test_store.py
git commit -m "feat(server): add in-memory upload and job stores"
```

---

### Task 6: server/main.py — FastAPI 엔드포인트

**Files:**
- Modify: `pyproject.toml:7-22` (fastapi, uvicorn, python-multipart, httpx 추가)
- Create: `server/main.py`
- Test: `tests/server/test_api.py`

- [ ] **Step 1: 의존성 추가**

pyproject.toml `dependencies`에 추가:

```toml
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "python-multipart>=0.0.9",
```

dev 그룹에 추가 (TestClient용):

```toml
    "httpx>=0.27.0",
```

Run: `uv sync`
Expected: 신규 패키지 설치 완료

- [ ] **Step 2: 실패하는 테스트 작성**

분석 함수는 monkeypatch로 대체해서 무거운 모델 없이 API 흐름만 검증한다:

```python
# tests/server/test_api.py
import io

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.main import app

CSV = "Date,User,Message\n2025-01-06 10:00:00,지언,안녕 뭐해?\n2025-01-06 10:01:00,민수,일하지\n"


@pytest.fixture
def client(sample_payload_dict, monkeypatch):
    # run_analysis와 직렬화를 가짜로 대체 (모델 로딩 회피)
    from server.schemas import ReportPayload

    def fake_run(df, opts, progress_cb=lambda i, l: None, **kw):
        progress_cb(0, "데이터 준비 중")
        return {"fake": True}

    monkeypatch.setattr(main, "run_analysis", fake_run)
    monkeypatch.setattr(
        main, "build_report_payload",
        lambda result: ReportPayload.model_validate(sample_payload_dict),
    )
    return TestClient(app)


def _upload(client):
    return client.post(
        "/api/upload",
        files={"file": ("chat.csv", io.BytesIO(CSV.encode("utf-8")), "text/csv")},
    )


def test_upload_returns_summary(client):
    res = _upload(client)
    assert res.status_code == 200
    body = res.json()
    assert set(body["users"]) == {"지언", "민수"}
    assert body["message_count"] == 2
    assert body["first_date"] == "2025-01-06"


def test_upload_invalid_csv_returns_400(client):
    res = client.post(
        "/api/upload",
        files={"file": ("bad.csv", io.BytesIO(b"Date,User,Message\n"), "text/csv")},
    )
    assert res.status_code == 400


def test_analyze_then_poll_job(client):
    upload_id = _upload(client).json()["upload_id"]
    res = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # TestClient는 BackgroundTask를 응답 후 동기 실행하므로 바로 done
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["result"]["me"] == "지언"


def test_analyze_unknown_upload_returns_404(client):
    res = client.post("/api/analyze", json={
        "upload_id": "nope", "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 404


def test_analyze_unknown_user_returns_400(client):
    upload_id = _upload(client).json()["upload_id"]
    res = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "없는사람",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    })
    assert res.status_code == 400


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_analysis_error_sets_job_error(client, monkeypatch):
    def boom(df, opts, progress_cb=lambda i, l: None, **kw):
        raise ValueError("기간에 메시지가 없습니다")

    monkeypatch.setattr(main, "run_analysis", boom)
    upload_id = _upload(client).json()["upload_id"]
    job_id = client.post("/api/analyze", json={
        "upload_id": upload_id, "me": "지언",
        "start_date": "2025-01-01", "end_date": "2025-12-31",
    }).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "error"
    assert "기간" in job["error"]
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/server/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: server.main`

- [ ] **Step 4: server/main.py 구현**

```python
# server/main.py
"""연애 권력 불균형 진단 API. 실행: uvicorn server.main:app --reload"""
import io
import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.kakao_parser import parse_kakao_chat
from server.analysis import AnalysisOptions, TOTAL_STEPS, run_analysis
from server.serialize import build_report_payload
from server.schemas import AnalyzeRequest, JobStatus, UploadSummary
from server.store import JobStore, UploadStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Love Imbalance Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

uploads = UploadStore()
jobs = JobStore()


@app.post("/api/upload", response_model=UploadSummary)
async def upload(file: UploadFile) -> UploadSummary:
    raw = await file.read()
    try:
        df = parse_kakao_chat(io.BytesIO(raw))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("CSV 파싱 실패")
        raise HTTPException(status_code=400, detail=f"CSV를 읽을 수 없습니다: {e}")

    return UploadSummary(
        upload_id=uploads.put(df),
        users=df["User"].unique().tolist(),
        message_count=len(df),
        first_date=df["Date"].min().date(),
        last_date=df["Date"].max().date(),
    )


def _run_job(job_id: str, df, opts: AnalysisOptions):
    try:
        result = run_analysis(
            df, opts,
            progress_cb=lambda i, label: jobs.set_progress(job_id, step=i, label=label),
        )
        jobs.set_done(job_id, build_report_payload(result))
    except ValueError as e:
        jobs.set_error(job_id, str(e))
    except Exception as e:
        logger.exception("분석 실패")
        jobs.set_error(job_id, f"분석 중 오류가 발생했습니다: {e}")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    df = uploads.get(req.upload_id)
    if df is None:
        raise HTTPException(status_code=404, detail="업로드를 찾을 수 없습니다. 다시 업로드해주세요.")
    if req.me not in df["User"].unique():
        raise HTTPException(status_code=400, detail=f"'{req.me}'는 대화 참여자가 아닙니다.")
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="시작일이 종료일보다 늦습니다.")

    opts = AnalysisOptions(
        me=req.me, start_date=req.start_date, end_date=req.end_date,
        session_gap=req.session_gap, preset=req.preset, api_key=req.api_key,
    )
    job_id = jobs.create(total_steps=TOTAL_STEPS)
    background_tasks.add_task(_run_job, job_id, df, opts)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없습니다.")
    return JobStatus(
        job_id=job.job_id, status=job.status, step=job.step,
        total_steps=job.total_steps, label=job.label,
        result=job.result, error=job.error,
    )
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/server/ -v`
Expected: 전체 PASS

- [ ] **Step 6: 서버 기동 스모크 테스트**

Run: `uv run uvicorn server.main:app --port 8000` (백그라운드) 후 `curl -s http://localhost:8000/docs | head -5` 확인, 서버 종료
Expected: Swagger UI HTML 응답

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml uv.lock server/main.py tests/server/test_api.py
git commit -m "feat(server): add FastAPI endpoints for upload, analyze, and job polling"
```

---

### Task 7: 프론트엔드 스캐폴드 + 디자인 토큰 시스템

**Files:**
- Create: `frontend/` (Vite react-ts 템플릿)
- Create: `frontend/src/styles/_tokens.scss`, `frontend/src/styles/global.scss`
- Modify: `frontend/vite.config.ts` (API 프록시 + vitest), `frontend/index.html` (폰트, 타이틀), `frontend/package.json` (test 스크립트)
- Delete: 템플릿 기본 파일 (`src/App.css`, `src/index.css`, `src/assets/react.svg`, `public/vite.svg`)

- [ ] **Step 1: Vite 스캐폴드 생성**

Run (프로젝트 루트에서, 백그라운드 권장):
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install && npm install sass recharts react-markdown
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```
Expected: frontend/ 생성, 의존성 설치 완료

- [ ] **Step 2: vite.config.ts에 프록시 + vitest 설정**

```typescript
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
```

package.json `scripts`에 `"test": "vitest run"` 추가.

- [ ] **Step 3: index.html — 폰트와 메타**

```html
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Relationship Audit — 연애 권력 불균형 진단</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600;700;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
      rel="stylesheet"
    />
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
    />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: 디자인 토큰 — src/styles/_tokens.scss**

```scss
// 에디토리얼 리포트 디자인 토큰
// 컨셉: 관계 감사 보고서 — 크림 종이 위 잉크, 시그널 레드 포인트

:root {
  // 컬러
  --paper: #f6f1e7;          // 크림 종이
  --paper-raised: #fdfbf4;   // 카드/박스 표면
  --ink: #1c1814;            // 본문 잉크
  --ink-soft: #6b6258;       // 보조 텍스트
  --hairline: rgba(28, 24, 20, 0.22);
  --hairline-strong: rgba(28, 24, 20, 0.55);
  --signal: #c8361f;         // 시그널 레드 (강조/불균형/상대)
  --signal-soft: rgba(200, 54, 31, 0.07);
  --me: #1f4d8f;             // 나 (잉크 블루)
  --me-soft: rgba(31, 77, 143, 0.08);
  --ok: #3d6b35;             // 균형/긍정 (딥 그린)

  // 타이포
  --font-display: 'Noto Serif KR', serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-body: 'Pretendard Variable', Pretendard, sans-serif;

  // 스페이싱 리듬 (8px 기반)
  --space-1: 0.5rem;
  --space-2: 1rem;
  --space-3: 1.5rem;
  --space-4: 2.5rem;
  --space-5: 4rem;
  --space-6: 6rem;

  --content-width: 56rem;    // 리포트 본문 폭
  --radius: 2px;             // 문서 느낌: 거의 직각
}

// 모노 라벨 (섹션 번호, 데이터 라벨용)
@mixin mono-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-soft);
}

// 헤어라인 위 테두리
@mixin rule-top($weight: 1px) {
  border-top: $weight solid var(--hairline-strong);
}

// 리포트 섹션 등장 모션 (스태거는 --stagger-i 인덱스로)
@mixin rise-in {
  opacity: 0;
  transform: translateY(14px);
  animation: rise-in 0.7s cubic-bezier(0.22, 1, 0.36, 1) forwards;
  animation-delay: calc(var(--stagger-i, 0) * 120ms);
}
```

- [ ] **Step 5: 글로벌 스타일 — src/styles/global.scss**

```scss
@use './tokens' as *;

*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
}

html {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-body);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

body {
  min-height: 100vh;
  // 종이 질감: 미세 노이즈 + 상단 비네트
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(28, 24, 20, 0.05), transparent),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
}

h1, h2, h3 {
  font-family: var(--font-display);
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.01em;
}

button {
  font: inherit;
  cursor: pointer;
}

@keyframes rise-in {
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

- [ ] **Step 6: 템플릿 정리**

- `src/App.css`, `src/index.css`, `src/assets/react.svg`, `public/vite.svg` 삭제
- `src/main.tsx`에서 `import './index.css'` → `import './styles/global.scss'`로 교체
- `src/App.tsx`는 임시로 비움:

```tsx
export default function App() {
  return <main>Relationship Audit</main>
}
```

- [ ] **Step 7: 빌드 확인**

Run: `cd frontend && npm run build`
Expected: `tsc -b && vite build` 성공

- [ ] **Step 8: .gitignore 확인 후 커밋**

루트 `.gitignore`에 `node_modules/`, `frontend/dist/` 없으면 추가.

```bash
git add frontend/ .gitignore
git commit -m "feat(frontend): scaffold vite react-ts app with editorial design tokens"
```

---

### Task 8: API 타입 + 클라이언트 + 포맷 유틸

**Files:**
- Create: `frontend/src/api/types.ts`, `frontend/src/api/client.ts`
- Create: `frontend/src/lib/format.ts`
- Test: `frontend/src/lib/format.test.ts`, `frontend/src/api/client.test.ts`

- [ ] **Step 1: 타입 정의 — src/api/types.ts** (server/schemas.py 미러)

```typescript
export interface UploadSummary {
  upload_id: string
  users: string[]
  message_count: number
  first_date: string
  last_date: string
}

export interface AnalyzeParams {
  upload_id: string
  me: string
  start_date: string
  end_date: string
  session_gap: number
  preset: string
  api_key?: string | null
}

export interface RadarPayload {
  categories: string[]
  me: number[]
  partner: number[]
}

export interface Participation {
  message_count_ratio: number
  char_count_ratio: number
  avg_length_me: number
  avg_length_partner: number
}

export interface TimelinePoint {
  week: string
  me: number
  partner: number
}

export type EmotionGroup = 'joy' | 'anger' | 'sadness' | 'anxiety' | 'hurt' | 'embarrass'

export interface EmotionPayload {
  me: Record<EmotionGroup, number>
  partner: Record<EmotionGroup, number>
  joy_gap: number
  negative_gap: number
}

export interface BoxStats {
  lo: number
  q1: number
  median: number
  q3: number
  hi: number
}

export interface ReplyTimePayload {
  me_median_sec: number
  partner_median_sec: number
  me_box: BoxStats
  partner_box: BoxStats
}

export interface QAPair {
  questioner: string
  question: string
  answerer: string
  answer: string
  score: number
}

export interface QASincerityPayload {
  avg_sincerity: number
  my_sincerity: number
  partner_sincerity: number
  pairs: QAPair[]
}

export interface AxisComparison {
  tier1: number
  llm: number
  delta: number
  agree: boolean
}

export interface EvidenceWindow {
  text: string
  sim: number
}

export interface LLMPayload {
  confidence: number
  report: string
  dominance: AxisComparison
  dependence: AxisComparison
  evidence: Record<string, EvidenceWindow[]>
}

export interface ReportPayload {
  me: string
  partner: string
  dominance_index: number
  dependence_index: number
  balance: number
  radar: RadarPayload
  participation: Participation
  timeline: TimelinePoint[]
  emotion: EmotionPayload
  reply_time: ReplyTimePayload
  double_text: { me: number; partner: number }
  initiation_ratio: number
  qa_sincerity: QASincerityPayload
  llm: LLMPayload | null
  llm_error: string | null
}

export type JobState = 'pending' | 'running' | 'done' | 'error'

export interface JobStatus {
  job_id: string
  status: JobState
  step: number
  total_steps: number
  label: string
  result: ReportPayload | null
  error: string | null
}
```

- [ ] **Step 2: 포맷 유틸 테스트 작성 — src/lib/format.test.ts**

```typescript
import { describe, expect, it } from 'vitest'
import { formatReplyTime, interpretIndex, pct } from './format'

describe('formatReplyTime', () => {
  it('formats seconds under a minute', () => {
    expect(formatReplyTime(35)).toBe('35초')
  })
  it('formats minutes under an hour', () => {
    expect(formatReplyTime(150)).toBe('2.5분')
  })
  it('drops trailing zero decimals', () => {
    expect(formatReplyTime(180)).toBe('3분')
  })
  it('formats hours', () => {
    expect(formatReplyTime(5400)).toBe('1.5시간')
  })
})

describe('interpretIndex', () => {
  it('me leads above 0.65', () => {
    expect(interpretIndex(0.7, '지언', '민수')).toEqual({ tone: 'me', text: '지언 쪽이 우위' })
  })
  it('partner leads below 0.35', () => {
    expect(interpretIndex(0.3, '지언', '민수')).toEqual({ tone: 'partner', text: '민수 쪽이 우위' })
  })
  it('balanced in between', () => {
    expect(interpretIndex(0.5, '지언', '민수')).toEqual({ tone: 'balanced', text: '균형' })
  })
})

describe('pct', () => {
  it('formats ratio as percent', () => {
    expect(pct(0.553)).toBe('55%')
    expect(pct(0.553, 1)).toBe('55.3%')
  })
})
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `cd frontend && npx vitest run src/lib/format.test.ts`
Expected: FAIL — format.ts 없음

- [ ] **Step 4: src/lib/format.ts 구현**

```typescript
export function formatReplyTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}초`
  const minutes = seconds / 60
  if (minutes < 60) return `${minutes.toFixed(1).replace(/\.0$/, '')}분`
  const hours = minutes / 60
  return `${hours.toFixed(1).replace(/\.0$/, '')}시간`
}

export type IndexTone = 'me' | 'partner' | 'balanced'

export interface IndexInterpretation {
  tone: IndexTone
  text: string
}

export function interpretIndex(value: number, me: string, partner: string): IndexInterpretation {
  if (value >= 0.65) return { tone: 'me', text: `${me} 쪽이 우위` }
  if (value <= 0.35) return { tone: 'partner', text: `${partner} 쪽이 우위` }
  return { tone: 'balanced', text: '균형' }
}

export function pct(ratio: number, digits = 0): string {
  return `${(ratio * 100).toFixed(digits)}%`
}
```

- [ ] **Step 5: API 클라이언트 테스트 — src/api/client.test.ts**

```typescript
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, uploadChat } from './client'

afterEach(() => vi.restoreAllMocks())

describe('uploadChat', () => {
  it('posts file and returns summary', async () => {
    const summary = { upload_id: 'u1', users: ['a', 'b'], message_count: 2, first_date: '2025-01-01', last_date: '2025-02-01' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(summary), { status: 200 }),
    ))
    const file = new File(['Date,User,Message'], 'chat.csv', { type: 'text/csv' })
    await expect(uploadChat(file)).resolves.toEqual(summary)
  })

  it('throws ApiError with server detail on 400', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: '분석 가능한 메시지가 없습니다.' }), { status: 400 }),
    ))
    const file = new File([''], 'bad.csv', { type: 'text/csv' })
    await expect(uploadChat(file)).rejects.toThrow('분석 가능한 메시지가 없습니다.')
  })
})
```

- [ ] **Step 6: src/api/client.ts 구현**

```typescript
import type { AnalyzeParams, JobStatus, UploadSummary } from './types'

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parseResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `요청 실패 (HTTP ${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // JSON이 아니면 기본 메시지 유지
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export async function uploadChat(file: File): Promise<UploadSummary> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/upload', { method: 'POST', body: form })
  return parseResponse<UploadSummary>(res)
}

export async function startAnalysis(params: AnalyzeParams): Promise<{ job_id: string }> {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  return parseResponse<{ job_id: string }>(res)
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${jobId}`)
  return parseResponse<JobStatus>(res)
}
```

- [ ] **Step 7: 테스트 통과 확인**

Run: `cd frontend && npm test`
Expected: format + client 테스트 모두 PASS

- [ ] **Step 8: 커밋**

```bash
git add frontend/src/api frontend/src/lib
git commit -m "feat(frontend): add API types, client, and formatting utilities"
```

---

### Task 9: 앱 상태 머신

**Files:**
- Create: `frontend/src/state/appState.ts`
- Test: `frontend/src/state/appState.test.ts`

(App.tsx 와이어링은 페이지 컴포넌트가 생기는 Task 10에서 함께 커밋 — 빌드 깨짐 방지)

- [ ] **Step 1: 리듀서 테스트 작성 — src/state/appState.test.ts**

```typescript
import { describe, expect, it } from 'vitest'
import { appReducer, initialState, type AppState } from './appState'
import type { ReportPayload, UploadSummary } from '../api/types'

const summary: UploadSummary = {
  upload_id: 'u1', users: ['지언', '민수'], message_count: 10,
  first_date: '2025-01-01', last_date: '2025-06-01',
}

describe('appReducer', () => {
  it('starts at upload phase', () => {
    expect(initialState.phase).toBe('upload')
  })

  it('UPLOADED moves to configure with summary', () => {
    const next = appReducer(initialState, { type: 'UPLOADED', summary })
    expect(next.phase).toBe('configure')
    expect(next.summary).toEqual(summary)
    expect(initialState.phase).toBe('upload') // 원본 불변
  })

  it('ANALYSIS_STARTED moves to analyzing with jobId', () => {
    const configured: AppState = { ...initialState, phase: 'configure', summary }
    const next = appReducer(configured, { type: 'ANALYSIS_STARTED', jobId: 'j1' })
    expect(next.phase).toBe('analyzing')
    expect(next.jobId).toBe('j1')
  })

  it('ANALYSIS_DONE moves to report with payload', () => {
    const report = { me: '지언' } as ReportPayload
    const next = appReducer(
      { ...initialState, phase: 'analyzing', jobId: 'j1' },
      { type: 'ANALYSIS_DONE', report },
    )
    expect(next.phase).toBe('report')
    expect(next.report?.me).toBe('지언')
  })

  it('FAILED keeps summary so user can retry config', () => {
    const next = appReducer(
      { ...initialState, phase: 'analyzing', summary, jobId: 'j1' },
      { type: 'FAILED', error: '기간에 메시지가 없습니다' },
    )
    expect(next.phase).toBe('configure')
    expect(next.error).toContain('기간')
    expect(next.summary).toEqual(summary)
  })

  it('RESET returns to initial state', () => {
    const next = appReducer({ ...initialState, phase: 'report' }, { type: 'RESET' })
    expect(next).toEqual(initialState)
  })
})
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd frontend && npx vitest run src/state`
Expected: FAIL — appState.ts 없음

- [ ] **Step 3: src/state/appState.ts 구현**

```typescript
import type { ReportPayload, UploadSummary } from '../api/types'

export type Phase = 'upload' | 'configure' | 'analyzing' | 'report'

export interface AppState {
  phase: Phase
  summary: UploadSummary | null
  jobId: string | null
  report: ReportPayload | null
  error: string | null
}

export type AppAction =
  | { type: 'UPLOADED'; summary: UploadSummary }
  | { type: 'ANALYSIS_STARTED'; jobId: string }
  | { type: 'ANALYSIS_DONE'; report: ReportPayload }
  | { type: 'FAILED'; error: string }
  | { type: 'RESET' }

export const initialState: AppState = {
  phase: 'upload',
  summary: null,
  jobId: null,
  report: null,
  error: null,
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'UPLOADED':
      return { ...state, phase: 'configure', summary: action.summary, error: null }
    case 'ANALYSIS_STARTED':
      return { ...state, phase: 'analyzing', jobId: action.jobId, error: null }
    case 'ANALYSIS_DONE':
      return { ...state, phase: 'report', report: action.report, error: null }
    case 'FAILED':
      // 설정 화면으로 돌려보내 재시도 가능하게 (업로드 요약은 보존)
      return { ...state, phase: state.summary ? 'configure' : 'upload', error: action.error }
    case 'RESET':
      return initialState
  }
}
```

- [ ] **Step 4: 테스트 통과 확인 후 커밋**

Run: `cd frontend && npm test`
Expected: PASS

```bash
git add frontend/src/state
git commit -m "feat(frontend): add app phase state machine"
```

---

### Task 10: UploadPage — 업로드 + 설정 폼 + App 와이어링

**Files:**
- Create: `frontend/src/components/UploadPage.tsx`, `frontend/src/components/UploadPage.module.scss`
- Create: `frontend/src/components/AnalyzingPage.tsx` (임시 스텁), `frontend/src/components/report/ReportPage.tsx` (임시 스텁)
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: UploadPage.tsx 구현**

업로드(upload phase)와 설정(configure phase)을 한 페이지에서 처리. 에디토리얼 마스트헤드 + 점선 드롭존 + 모노 데이터 요약 + 설정 폼:

```tsx
import { useRef, useState, type Dispatch } from 'react'
import { startAnalysis, uploadChat } from '../api/client'
import type { AppAction, AppState } from '../state/appState'
import styles from './UploadPage.module.scss'

const PRESETS = ['기본', '답장속도 중시', '감정 중시']

interface Props {
  state: AppState
  dispatch: Dispatch<AppAction>
}

export function UploadPage({ state, dispatch }: Props) {
  const { summary, error } = state
  const fileInput = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [me, setMe] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [preset, setPreset] = useState(PRESETS[0])
  const [sessionGap, setSessionGap] = useState(30)
  const [apiKey, setApiKey] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)

  const handleFile = async (file: File) => {
    setBusy(true)
    setLocalError(null)
    try {
      const uploaded = await uploadChat(file)
      setMe(uploaded.users[0])
      setStartDate(uploaded.first_date)
      setEndDate(uploaded.last_date)
      dispatch({ type: 'UPLOADED', summary: uploaded })
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '업로드에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  const handleAnalyze = async () => {
    if (!summary) return
    if (startDate > endDate) {
      setLocalError('시작일이 종료일보다 늦습니다.')
      return
    }
    setBusy(true)
    setLocalError(null)
    try {
      const { job_id } = await startAnalysis({
        upload_id: summary.upload_id,
        me,
        start_date: startDate,
        end_date: endDate,
        session_gap: sessionGap,
        preset,
        api_key: apiKey || null,
      })
      dispatch({ type: 'ANALYSIS_STARTED', jobId: job_id })
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '분석 시작에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.masthead}>
        <p className={styles.kicker}>Relationship Audit</p>
        <h1 className={styles.title}>
          당신과 그 사람,
          <br />
          누가 더 <em>기울어져</em> 있나요?
        </h1>
        <p className={styles.lede}>
          카카오톡 대화를 AI가 읽고, 관계의 권력 불균형을 진단합니다.
          데이터는 분석에만 쓰이고 저장되지 않습니다.
        </p>
      </header>

      {!summary && (
        <section
          className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            const file = e.dataTransfer.files[0]
            if (file) void handleFile(file)
          }}
          onClick={() => fileInput.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && fileInput.current?.click()}
        >
          <input
            ref={fileInput}
            type="file"
            accept=".csv"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void handleFile(file)
            }}
          />
          <p className={styles.dropLabel}>{busy ? '읽는 중…' : 'CSV 파일을 끌어다 놓거나 클릭'}</p>
          <p className={styles.dropHint}>카카오톡 PC → 대화방 메뉴 → 대화 내보내기 → CSV</p>
        </section>
      )}

      {summary && (
        <section className={styles.configure}>
          <dl className={styles.summaryRow}>
            <div>
              <dt>총 메시지</dt>
              <dd>{summary.message_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>첫 대화</dt>
              <dd>{summary.first_date}</dd>
            </div>
            <div>
              <dt>마지막 대화</dt>
              <dd>{summary.last_date}</dd>
            </div>
          </dl>

          <div className={styles.formGrid}>
            <label>
              <span>나는 누구인가요</span>
              <select value={me} onChange={(e) => setMe(e.target.value)}>
                {summary.users.map((u) => (
                  <option key={u} value={u}>{u}</option>
                ))}
              </select>
            </label>
            <label>
              <span>가중치 프리셋</span>
              <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                {PRESETS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </label>
            <label>
              <span>분석 시작일</span>
              <input type="date" value={startDate} min={summary.first_date} max={summary.last_date}
                onChange={(e) => setStartDate(e.target.value)} />
            </label>
            <label>
              <span>분석 종료일</span>
              <input type="date" value={endDate} min={summary.first_date} max={summary.last_date}
                onChange={(e) => setEndDate(e.target.value)} />
            </label>
            <label>
              <span>세션 구분 간격 — {sessionGap}분</span>
              <input type="range" min={10} max={120} step={5} value={sessionGap}
                onChange={(e) => setSessionGap(Number(e.target.value))} />
            </label>
            <label>
              <span>OpenAI API 키 (선택 — AI 심층 분석)</span>
              <input type="password" value={apiKey} placeholder="sk-…"
                onChange={(e) => setApiKey(e.target.value)} autoComplete="off" />
            </label>
          </div>

          <button className={styles.cta} onClick={() => void handleAnalyze()} disabled={busy}>
            {busy ? '시작 중…' : '감사 시작 →'}
          </button>
          <button className={styles.resetLink} onClick={() => dispatch({ type: 'RESET' })}>
            다른 파일 업로드
          </button>
        </section>
      )}

      {(localError ?? error) && <p className={styles.error}>{localError ?? error}</p>}
    </main>
  )
}
```

- [ ] **Step 2: UploadPage.module.scss 구현**

```scss
@use '../styles/tokens' as *;

.page {
  max-width: var(--content-width);
  margin: 0 auto;
  padding: var(--space-6) var(--space-3) var(--space-5);
}

.masthead {
  @include rule-top(3px);
  padding-top: var(--space-2);
  margin-bottom: var(--space-5);
}

.kicker {
  @include mono-label;
  color: var(--signal);
  margin-bottom: var(--space-3);
}

.title {
  font-size: clamp(2.2rem, 6vw, 4rem);
  font-weight: 900;
  margin-bottom: var(--space-3);

  em {
    font-style: italic;
    color: var(--signal);
  }
}

.lede {
  max-width: 32rem;
  color: var(--ink-soft);
}

.dropzone {
  border: 1.5px dashed var(--hairline-strong);
  background: var(--paper-raised);
  padding: var(--space-5) var(--space-3);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;

  &:hover,
  &.dragOver {
    border-color: var(--signal);
    background: var(--signal-soft);
  }
}

.dropLabel {
  font-family: var(--font-display);
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: var(--space-1);
}

.dropHint {
  @include mono-label;
}

.configure {
  @include rule-top;
  padding-top: var(--space-3);
}

.summaryRow {
  display: flex;
  gap: var(--space-5);
  margin-bottom: var(--space-4);

  dt {
    @include mono-label;
    margin-bottom: 0.25rem;
  }

  dd {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 700;
  }
}

.formGrid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-3);
  margin-bottom: var(--space-4);

  label {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;

    span {
      @include mono-label;
    }

    select,
    input[type='date'],
    input[type='password'] {
      font: inherit;
      font-family: var(--font-mono);
      font-size: 0.9rem;
      padding: 0.6rem 0.75rem;
      border: 1px solid var(--hairline-strong);
      border-radius: var(--radius);
      background: var(--paper-raised);
      color: var(--ink);

      &:focus {
        outline: 2px solid var(--me);
        outline-offset: 1px;
      }
    }

    input[type='range'] {
      accent-color: var(--signal);
    }
  }
}

.cta {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  padding: 0.9rem 2.5rem;
  background: var(--ink);
  color: var(--paper);
  border: none;
  border-radius: var(--radius);
  transition: background 0.2s;

  &:hover:not(:disabled) {
    background: var(--signal);
  }

  &:disabled {
    opacity: 0.5;
    cursor: wait;
  }
}

.resetLink {
  @include mono-label;
  background: none;
  border: none;
  margin-left: var(--space-3);
  text-decoration: underline;
}

.error {
  margin-top: var(--space-3);
  padding: var(--space-2);
  border-left: 3px solid var(--signal);
  background: var(--signal-soft);
  color: var(--signal);
}

@media (max-width: 640px) {
  .formGrid {
    grid-template-columns: 1fr;
  }

  .summaryRow {
    flex-direction: column;
    gap: var(--space-2);
  }
}
```

- [ ] **Step 3: App.tsx 와이어링 + 임시 스텁 생성**

```tsx
// frontend/src/App.tsx
import { useReducer } from 'react'
import { appReducer, initialState } from './state/appState'
import { UploadPage } from './components/UploadPage'
import { AnalyzingPage } from './components/AnalyzingPage'
import { ReportPage } from './components/report/ReportPage'

export default function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)

  if (state.phase === 'analyzing' && state.jobId) {
    return <AnalyzingPage jobId={state.jobId} dispatch={dispatch} />
  }
  if (state.phase === 'report' && state.report) {
    return <ReportPage report={state.report} onReset={() => dispatch({ type: 'RESET' })} />
  }
  return <UploadPage state={state} dispatch={dispatch} />
}
```

```tsx
// frontend/src/components/AnalyzingPage.tsx (임시 — Task 11에서 교체)
import type { Dispatch } from 'react'
import type { AppAction } from '../state/appState'

export function AnalyzingPage(_props: { jobId: string; dispatch: Dispatch<AppAction> }) {
  return <main>분석 중…</main>
}
```

```tsx
// frontend/src/components/report/ReportPage.tsx (임시 — Task 12에서 교체)
import type { ReportPayload } from '../../api/types'

export function ReportPage(_props: { report: ReportPayload; onReset: () => void }) {
  return <main>리포트</main>
}
```

- [ ] **Step 4: 빌드 + 수동 확인**

Run: `cd frontend && npm run build`
Expected: 성공

Run (수동 스모크): 터미널 1 `uv run uvicorn server.main:app --port 8000`, 터미널 2 `cd frontend && npm run dev` → 브라우저에서 CSV 업로드 → 요약/폼 표시 확인

- [ ] **Step 5: 커밋**

```bash
git add frontend/src
git commit -m "feat(frontend): add editorial upload and configure page"
```

---

### Task 11: AnalyzingPage — 잡 폴링 + 진행 연출

**Files:**
- Replace: `frontend/src/components/AnalyzingPage.tsx`
- Create: `frontend/src/components/AnalyzingPage.module.scss`

- [ ] **Step 1: AnalyzingPage.tsx 구현**

1.5초 간격 폴링. 에디토리얼 연출: 큰 세리프 현재 단계 + 시그널 레드 진행선 + 모노 카운터:

```tsx
import { useEffect, useRef, useState, type Dispatch } from 'react'
import { getJob } from '../api/client'
import type { AppAction } from '../state/appState'
import styles from './AnalyzingPage.module.scss'

const POLL_MS = 1500

interface Props {
  jobId: string
  dispatch: Dispatch<AppAction>
}

export function AnalyzingPage({ jobId, dispatch }: Props) {
  const [step, setStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(7)
  const [label, setLabel] = useState('대기 중')
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false

    const poll = async () => {
      try {
        const job = await getJob(jobId)
        if (stopped.current) return
        setStep(job.step)
        setTotalSteps(job.total_steps)
        setLabel(job.label)
        if (job.status === 'done' && job.result) {
          dispatch({ type: 'ANALYSIS_DONE', report: job.result })
          return
        }
        if (job.status === 'error') {
          dispatch({ type: 'FAILED', error: job.error ?? '분석에 실패했습니다.' })
          return
        }
        window.setTimeout(() => void poll(), POLL_MS)
      } catch (e) {
        if (stopped.current) return
        dispatch({ type: 'FAILED', error: e instanceof Error ? e.message : '연결이 끊겼습니다.' })
      }
    }

    void poll()
    return () => {
      stopped.current = true
    }
  }, [jobId, dispatch])

  const progress = totalSteps > 0 ? (step + 1) / totalSteps : 0

  return (
    <main className={styles.page}>
      <p className={styles.kicker}>Audit in progress</p>
      <h1 className={styles.label} key={label}>
        {label}
        <span className={styles.ellipsis} aria-hidden>…</span>
      </h1>
      <div className={styles.track} role="progressbar"
        aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className={styles.fill} style={{ width: `${progress * 100}%` }} />
      </div>
      <p className={styles.counter}>
        {String(step + 1).padStart(2, '0')} / {String(totalSteps).padStart(2, '0')}
      </p>
    </main>
  )
}
```

- [ ] **Step 2: AnalyzingPage.module.scss 구현**

```scss
@use '../styles/tokens' as *;

.page {
  max-width: var(--content-width);
  margin: 0 auto;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: var(--space-3);
}

.kicker {
  @include mono-label;
  color: var(--signal);
  margin-bottom: var(--space-3);
}

.label {
  font-size: clamp(2rem, 5vw, 3.2rem);
  margin-bottom: var(--space-4);
  animation: fade-swap 0.5s ease;
}

.ellipsis {
  display: inline-block;
  animation: pulse 1.2s ease-in-out infinite;
}

.track {
  height: 2px;
  background: var(--hairline);
  margin-bottom: var(--space-2);
}

.fill {
  height: 100%;
  background: var(--signal);
  transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

.counter {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--ink-soft);
}

@keyframes fade-swap {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse {
  50% {
    opacity: 0.2;
  }
}
```

- [ ] **Step 3: 빌드 확인 + 커밋**

Run: `cd frontend && npm run build`
Expected: 성공

```bash
git add frontend/src/components/AnalyzingPage.tsx frontend/src/components/AnalyzingPage.module.scss
git commit -m "feat(frontend): add analyzing page with job polling and progress staging"
```

---

### Task 12: ReportPage 골격 — 마스트헤드 + 평결(Verdict) 섹션

**Files:**
- Replace: `frontend/src/components/report/ReportPage.tsx`
- Create: `frontend/src/components/report/ReportPage.module.scss`
- Create: `frontend/src/components/report/Section.tsx`, `frontend/src/components/report/Section.module.scss`
- Create: `frontend/src/components/report/VerdictSection.tsx`, `frontend/src/components/report/VerdictSection.module.scss`
- Create: `frontend/src/components/report/fixtureReport.ts`
- Test: `frontend/src/components/report/ReportPage.test.tsx`

- [ ] **Step 1: 렌더 스모크 테스트 + 픽스처 작성**

```tsx
// frontend/src/components/report/ReportPage.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ReportPage } from './ReportPage'
import { fixtureReport } from './fixtureReport'

describe('ReportPage', () => {
  it('renders verdict numbers and names', () => {
    render(<ReportPage report={fixtureReport} onReset={vi.fn()} />)
    expect(screen.getByText('0.62')).toBeTruthy()  // 지배성
    expect(screen.getByText('0.71')).toBeTruthy()  // 의존도
    expect(screen.getAllByText(/지언/).length).toBeGreaterThan(0)
  })

  it('shows llm nudge when llm is null', () => {
    render(<ReportPage report={fixtureReport} onReset={vi.fn()} />)
    expect(screen.getByText(/API 키를 입력하면/)).toBeTruthy()
  })
})
```

```typescript
// frontend/src/components/report/fixtureReport.ts
import type { ReportPayload } from '../../api/types'

const box = { lo: 0.2, q1: 0.5, median: 1.0, q3: 2.5, hi: 12.0 }

export const fixtureReport: ReportPayload = {
  me: '지언', partner: '민수',
  dominance_index: 0.62, dependence_index: 0.71, balance: 0.91,
  radar: {
    categories: ['선톡 비율', '대화 종료', '메시지 비율', '글자 비율', '답장 속도', '더블텍스트', 'QA 성의도'],
    me: [0.7, 0.5, 0.55, 0.6, 0.72, 0.4, 0.55],
    partner: [0.3, 0.5, 0.45, 0.4, 0.28, 0.6, 0.45],
  },
  participation: { message_count_ratio: 0.55, char_count_ratio: 0.6, avg_length_me: 18.2, avg_length_partner: 12.1 },
  timeline: [
    { week: '2025-01-06', me: 12, partner: 18 },
    { week: '2025-01-13', me: 30, partner: 22 },
    { week: '2025-01-20', me: 25, partner: 28 },
  ],
  emotion: {
    me: { joy: 0.42, anger: 0.08, sadness: 0.12, anxiety: 0.18, hurt: 0.1, embarrass: 0.1 },
    partner: { joy: 0.51, anger: 0.05, sadness: 0.1, anxiety: 0.14, hurt: 0.1, embarrass: 0.1 },
    joy_gap: -0.09, negative_gap: 0.09,
  },
  reply_time: { me_median_sec: 35, partner_median_sec: 420, me_box: box, partner_box: { ...box, median: 7, q3: 15, hi: 40 } },
  double_text: { me: 0.21, partner: 0.08 },
  initiation_ratio: 0.7,
  qa_sincerity: {
    avg_sincerity: 0.55, my_sincerity: 0.61, partner_sincerity: 0.49,
    pairs: [
      { questioner: '지언', question: '주말에 뭐할까?', answerer: '민수', answer: 'ㅇㅇ', score: 0.21 },
      { questioner: '민수', question: '저녁 먹었어?', answerer: '지언', answer: '응! 너 좋아하는 파스타 해먹었어', score: 0.78 },
    ],
  },
  llm: null,
  llm_error: null,
}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd frontend && npx vitest run src/components/report`
Expected: FAIL (스텁엔 해당 텍스트 없음)

- [ ] **Step 3: 공용 Section 컴포넌트**

```tsx
// frontend/src/components/report/Section.tsx
import type { CSSProperties, ReactNode } from 'react'
import styles from './Section.module.scss'

interface Props {
  no: string        // "01"
  title: string     // "대화량"
  staggerIndex: number
  children: ReactNode
}

export function Section({ no, title, staggerIndex, children }: Props) {
  return (
    <section className={styles.section} style={{ '--stagger-i': staggerIndex } as CSSProperties}>
      <header className={styles.header}>
        <span className={styles.no}>{no}</span>
        <h2 className={styles.title}>{title}</h2>
      </header>
      {children}
    </section>
  )
}
```

```scss
// frontend/src/components/report/Section.module.scss
@use '../../styles/tokens' as *;

.section {
  @include rule-top;
  @include rise-in;
  padding-top: var(--space-3);
  margin-bottom: var(--space-5);
}

.header {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}

.no {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--signal);
}

.title {
  font-size: 1.6rem;
}
```

- [ ] **Step 4: VerdictSection**

```tsx
// frontend/src/components/report/VerdictSection.tsx
import type { CSSProperties } from 'react'
import type { ReportPayload } from '../../api/types'
import { interpretIndex, type IndexTone } from '../../lib/format'
import styles from './VerdictSection.module.scss'

const TONE_CLASS: Record<IndexTone, string> = {
  me: styles.toneMe,
  partner: styles.tonePartner,
  balanced: styles.toneBalanced,
}

interface Props {
  report: ReportPayload
}

export function VerdictSection({ report }: Props) {
  const { me, partner, dominance_index, dependence_index, balance } = report
  const items = [
    { label: '지배성 지수', value: dominance_index, note: interpretIndex(dominance_index, me, partner), hint: '1에 가까울수록 내가 대화 주도' },
    { label: '의존도 지수', value: dependence_index, note: interpretIndex(dependence_index, me, partner), hint: '1에 가까울수록 내가 더 의존적' },
    {
      label: '균형 점수', value: balance,
      note: balance >= 0.7
        ? { tone: 'balanced' as const, text: '균형적인 관계' }
        : { tone: 'partner' as const, text: '불균형 감지' },
      hint: '1에 가까울수록 균형',
    },
  ]

  return (
    <div className={styles.verdict} style={{ '--stagger-i': 1 } as CSSProperties}>
      {items.map((item) => (
        <div className={styles.item} key={item.label}>
          <p className={styles.label}>{item.label}</p>
          <p className={`${styles.value} ${TONE_CLASS[item.note.tone]}`}>{item.value.toFixed(2)}</p>
          <p className={`${styles.note} ${TONE_CLASS[item.note.tone]}`}>{item.note.text}</p>
          <p className={styles.hint}>{item.hint}</p>
        </div>
      ))}
    </div>
  )
}
```

```scss
// frontend/src/components/report/VerdictSection.module.scss
@use '../../styles/tokens' as *;

.verdict {
  @include rise-in;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border-top: 3px solid var(--ink);
  margin-bottom: var(--space-5);
}

.item {
  padding: var(--space-3) var(--space-2) var(--space-3) 0;

  & + & {
    border-left: 1px solid var(--hairline);
    padding-left: var(--space-3);
  }
}

.label {
  @include mono-label;
  margin-bottom: var(--space-2);
}

.value {
  font-family: var(--font-display);
  font-size: clamp(2.6rem, 7vw, 4.2rem);
  font-weight: 900;
  line-height: 1;
  margin-bottom: var(--space-1);
}

.note {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.hint {
  font-size: 0.8rem;
  color: var(--ink-soft);
}

.toneMe { color: var(--me); }
.tonePartner { color: var(--signal); }
.toneBalanced { color: var(--ok); }

@media (max-width: 640px) {
  .verdict {
    grid-template-columns: 1fr;
  }

  .item + .item {
    border-left: none;
    border-top: 1px solid var(--hairline);
    padding-left: 0;
  }
}
```

- [ ] **Step 5: ReportPage.tsx — 골격 + 마스트헤드 + LLM 안내(임시)**

```tsx
// frontend/src/components/report/ReportPage.tsx
import type { CSSProperties } from 'react'
import type { ReportPayload } from '../../api/types'
import { Section } from './Section'
import { VerdictSection } from './VerdictSection'
import styles from './ReportPage.module.scss'

interface Props {
  report: ReportPayload
  onReset: () => void
}

export function ReportPage({ report, onReset }: Props) {
  const { me, partner } = report

  return (
    <main className={styles.page}>
      <header className={styles.masthead} style={{ '--stagger-i': 0 } as CSSProperties}>
        <div className={styles.mastRow}>
          <p className={styles.kicker}>Relationship Audit — Final Report</p>
          <button className={styles.reset} onClick={onReset}>새 분석 ↺</button>
        </div>
        <h1 className={styles.title}>
          {me} <span className={styles.vs}>&times;</span> {partner}
        </h1>
        <p className={styles.sub}>대화 권력 불균형 진단 결과</p>
      </header>

      <VerdictSection report={report} />

      {/* Task 13~15에서 섹션 추가: RadarSection, VolumeSection, EmotionSection,
          ReplySection, SinceritySection, AISection */}

      {report.llm === null && report.llm_error === null && (
        <Section no="—" title="AI 심층 분석" staggerIndex={6}>
          <p className={styles.llmNudge}>
            OpenAI API 키를 입력하면 LLM이 실제 대화 장면을 인용하며 심층 진단을 제공합니다.
            (규칙 기반 분석 결과는 위에 그대로 유지됩니다)
          </p>
        </Section>
      )}
    </main>
  )
}
```

```scss
// frontend/src/components/report/ReportPage.module.scss
@use '../../styles/tokens' as *;

.page {
  max-width: var(--content-width);
  margin: 0 auto;
  padding: var(--space-5) var(--space-3) var(--space-6);
}

.masthead {
  @include rise-in;
  border-top: 3px solid var(--ink);
  padding-top: var(--space-2);
  margin-bottom: var(--space-4);
}

.mastRow {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: var(--space-3);
}

.kicker {
  @include mono-label;
  color: var(--signal);
}

.reset {
  @include mono-label;
  background: none;
  border: 1px solid var(--hairline-strong);
  padding: 0.4rem 0.9rem;
  border-radius: var(--radius);
  transition: all 0.2s;

  &:hover {
    background: var(--ink);
    color: var(--paper);
  }
}

.title {
  font-size: clamp(2.4rem, 7vw, 4.5rem);
  font-weight: 900;
}

.vs {
  color: var(--signal);
  font-weight: 400;
}

.sub {
  color: var(--ink-soft);
  margin-top: var(--space-1);
}

.llmNudge {
  color: var(--ink-soft);
  background: var(--paper-raised);
  border: 1px dashed var(--hairline-strong);
  padding: var(--space-3);
}
```

- [ ] **Step 6: 테스트 통과 + 빌드 확인**

Run: `cd frontend && npm test && npm run build`
Expected: PASS + 빌드 성공

- [ ] **Step 7: 커밋**

```bash
git add frontend/src/components/report
git commit -m "feat(frontend): add report masthead and verdict section"
```

---

### Task 13: RadarSection + VolumeSection (01 대화량)

**Files:**
- Create: `frontend/src/components/report/chartTheme.ts`
- Create: `frontend/src/components/report/RadarSection.tsx`, `RadarSection.module.scss`
- Create: `frontend/src/components/report/VolumeSection.tsx`, `VolumeSection.module.scss`
- Modify: `frontend/src/components/report/ReportPage.tsx` (섹션 추가)

- [ ] **Step 1: 차트 공통 테마 — chartTheme.ts**

```typescript
// Recharts에 줄 공통 색/스타일. CSS 변수와 동일 값 (SVG 속성엔 변수 사용이 불안정한 곳 대비)
export const CHART = {
  me: '#1f4d8f',
  partner: '#c8361f',
  ink: '#1c1814',
  inkSoft: '#6b6258',
  hairline: 'rgba(28, 24, 20, 0.22)',
  mono: "'IBM Plex Mono', monospace",
} as const

export const monoTick = { fontFamily: CHART.mono, fontSize: 11, fill: CHART.inkSoft }
```

- [ ] **Step 2: RadarSection.tsx**

```tsx
import {
  Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer,
} from 'recharts'
import type { ReportPayload } from '../../api/types'
import { Section } from './Section'
import { CHART, monoTick } from './chartTheme'
import styles from './RadarSection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

export function RadarSection({ report, staggerIndex }: Props) {
  const { radar, me, partner } = report
  const data = radar.categories.map((category, i) => ({
    category,
    me: radar.me[i],
    partner: radar.partner[i],
  }))

  return (
    <Section no="00" title="대화 권력 지도" staggerIndex={staggerIndex}>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={380}>
          <RadarChart data={data} outerRadius="75%">
            <PolarGrid stroke={CHART.hairline} />
            <PolarAngleAxis dataKey="category" tick={monoTick} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
            <Radar name={me} dataKey="me" stroke={CHART.me} fill={CHART.me} fillOpacity={0.22} strokeWidth={2} />
            <Radar name={partner} dataKey="partner" stroke={CHART.partner} fill={CHART.partner} fillOpacity={0.22} strokeWidth={2} />
            <Legend wrapperStyle={{ fontFamily: CHART.mono, fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <p className={styles.caption}>7개 축 모두 0–1 정규화. 바깥쪽일수록 해당 축에서 그 사람의 비중이 큼.</p>
    </Section>
  )
}
```

```scss
// RadarSection.module.scss
@use '../../styles/tokens' as *;

.chartWrap {
  background: var(--paper-raised);
  border: 1px solid var(--hairline);
  padding: var(--space-2);
}

.caption {
  @include mono-label;
  text-transform: none;
  letter-spacing: 0.02em;
  margin-top: var(--space-1);
}
```

- [ ] **Step 3: VolumeSection.tsx (01 — 대화량)**

```tsx
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import { CHART, monoTick } from './chartTheme'
import styles from './VolumeSection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

export function VolumeSection({ report, staggerIndex }: Props) {
  const { me, partner, timeline, participation: p } = report
  const stats = [
    { label: `${me} 메시지 비율`, value: pct(p.message_count_ratio, 1) },
    { label: `${partner} 메시지 비율`, value: pct(1 - p.message_count_ratio, 1) },
    { label: `${me} 평균 길이`, value: `${Math.round(p.avg_length_me)}자` },
    { label: `${partner} 평균 길이`, value: `${Math.round(p.avg_length_partner)}자` },
  ]

  return (
    <Section no="01" title="대화량" staggerIndex={staggerIndex}>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={timeline} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid stroke={CHART.hairline} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="week" tick={monoTick} tickLine={false} axisLine={{ stroke: CHART.hairline }} />
            <YAxis tick={monoTick} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ fontFamily: CHART.mono, fontSize: 12, background: '#fdfbf4', border: `1px solid ${CHART.hairline}` }} />
            <Legend wrapperStyle={{ fontFamily: CHART.mono, fontSize: 12 }} />
            <Line type="monotone" dataKey="me" name={me} stroke={CHART.me} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="partner" name={partner} stroke={CHART.partner} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <dl className={styles.stats}>
        {stats.map((s) => (
          <div key={s.label}>
            <dt>{s.label}</dt>
            <dd>{s.value}</dd>
          </div>
        ))}
      </dl>
    </Section>
  )
}
```

```scss
// VolumeSection.module.scss
@use '../../styles/tokens' as *;

.chartWrap {
  background: var(--paper-raised);
  border: 1px solid var(--hairline);
  padding: var(--space-2);
  margin-bottom: var(--space-3);
}

.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-2);

  dt {
    @include mono-label;
    margin-bottom: 0.25rem;
  }

  dd {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 700;
  }

  @media (max-width: 640px) {
    grid-template-columns: repeat(2, 1fr);
  }
}
```

- [ ] **Step 4: ReportPage에 두 섹션 추가**

`<VerdictSection report={report} />` 아래에:

```tsx
<RadarSection report={report} staggerIndex={2} />
<VolumeSection report={report} staggerIndex={3} />
```

(상단에 `import { RadarSection } from './RadarSection'`, `import { VolumeSection } from './VolumeSection'` 추가)

- [ ] **Step 5: 테스트/빌드 + 커밋**

Run: `cd frontend && npm test && npm run build`
Expected: PASS

```bash
git add frontend/src/components/report
git commit -m "feat(frontend): add radar map and conversation volume sections"
```

---

### Task 14: EmotionSection (02) + ReplySection (03)

**Files:**
- Create: `frontend/src/components/report/EmotionSection.tsx`, `EmotionSection.module.scss`
- Create: `frontend/src/components/report/ReplySection.tsx`, `ReplySection.module.scss`
- Modify: `frontend/src/components/report/ReportPage.tsx`

- [ ] **Step 1: EmotionSection.tsx**

감정 6그룹의 가로 누적 막대(커스텀 CSS — 에디토리얼 톤 유지) + 불균형 경고 배너:

```tsx
import type { EmotionGroup, ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './EmotionSection.module.scss'

const EMOTIONS: { group: EmotionGroup; label: string; color: string }[] = [
  { group: 'joy', label: '기쁨', color: '#b58900' },
  { group: 'anger', label: '분노', color: '#c8361f' },
  { group: 'sadness', label: '슬픔', color: '#1f4d8f' },
  { group: 'anxiety', label: '불안', color: '#a45a1c' },
  { group: 'hurt', label: '상처', color: '#6b4d8f' },
  { group: 'embarrass', label: '당황', color: '#2a7d6f' },
]

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function StackedBar({ name, dist }: { name: string; dist: Record<EmotionGroup, number> }) {
  return (
    <div className={styles.barRow}>
      <span className={styles.barName}>{name}</span>
      <div className={styles.bar}>
        {EMOTIONS.map(({ group, label, color }) => {
          const ratio = dist[group] ?? 0
          if (ratio <= 0) return null
          return (
            <span
              key={group}
              className={styles.seg}
              style={{ width: `${ratio * 100}%`, background: color }}
              title={`${label} ${pct(ratio)}`}
            />
          )
        })}
      </div>
    </div>
  )
}

export function EmotionSection({ report, staggerIndex }: Props) {
  const { me, partner, emotion } = report
  const warning = emotion.negative_gap > 0.1

  return (
    <Section no="02" title="감정" staggerIndex={staggerIndex}>
      {warning && (
        <p className={styles.warning}>
          <strong>부정 감정 불균형 감지</strong> — {me}님이 부정적인 감정을 더 많이 표현하고 있어요.
          대화에서 더 많은 공감과 이해가 필요할 수 있습니다.
        </p>
      )}

      <div className={styles.bars}>
        <StackedBar name={me} dist={emotion.me} />
        <StackedBar name={partner} dist={emotion.partner} />
      </div>

      <ul className={styles.legend}>
        {EMOTIONS.map(({ group, label, color }) => (
          <li key={group}>
            <span className={styles.dot} style={{ background: color }} />
            {label}
            <span className={styles.legendVals}>
              {pct(emotion.me[group] ?? 0)} / {pct(emotion.partner[group] ?? 0)}
            </span>
          </li>
        ))}
      </ul>
      <p className={styles.caption}>범례 수치: {me} / {partner}</p>
    </Section>
  )
}
```

```scss
// EmotionSection.module.scss
@use '../../styles/tokens' as *;

.warning {
  border-left: 3px solid var(--signal);
  background: var(--signal-soft);
  padding: var(--space-2) var(--space-3);
  margin-bottom: var(--space-3);

  strong {
    color: var(--signal);
  }
}

.bars {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}

.barRow {
  display: grid;
  grid-template-columns: 6rem 1fr;
  align-items: center;
  gap: var(--space-2);
}

.barName {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  text-align: right;
}

.bar {
  display: flex;
  height: 1.6rem;
  background: var(--paper-raised);
  border: 1px solid var(--hairline);
  overflow: hidden;
}

.seg {
  display: block;
  height: 100%;
  transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-3);
  list-style: none;
  padding: 0;

  li {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
}

.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.legendVals {
  color: var(--ink-soft);
}

.caption {
  @include mono-label;
  margin-top: var(--space-1);
}
```

- [ ] **Step 2: ReplySection.tsx**

답장 중앙값 2장 카드 + 커스텀 박스 플롯(CSS) + 더블텍스트/선톡 비율 바:

```tsx
import type { BoxStats, ReportPayload } from '../../api/types'
import { formatReplyTime, pct } from '../../lib/format'
import { Section } from './Section'
import styles from './ReplySection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function BoxPlot({ name, box, max, color }: { name: string; box: BoxStats; max: number; color: string }) {
  const x = (v: number) => `${(v / max) * 100}%`
  const w = (a: number, b: number) => `${((b - a) / max) * 100}%`
  return (
    <div className={styles.boxRow}>
      <span className={styles.boxName}>{name}</span>
      <div className={styles.boxTrack}>
        <span className={styles.whisker} style={{ left: x(box.lo), width: w(box.lo, box.hi) }} />
        <span className={styles.iqr} style={{ left: x(box.q1), width: w(box.q1, box.q3), borderColor: color }} />
        <span className={styles.median} style={{ left: x(box.median), background: color }} />
      </div>
      <span className={styles.boxVal}>{box.median.toFixed(1)}분</span>
    </div>
  )
}

function RatioBar({ label, me, partner, meName, partnerName }: {
  label: string; me: number; partner: number; meName: string; partnerName: string
}) {
  const total = me + partner || 1
  return (
    <div className={styles.ratioCard}>
      <p className={styles.ratioTitle}>{label}</p>
      {[{ name: meName, v: me, cls: styles.fillMe }, { name: partnerName, v: partner, cls: styles.fillPartner }].map((row) => (
        <div key={row.name} className={styles.ratioRow}>
          <span>{row.name}</span>
          <div className={styles.ratioTrack}>
            <span className={`${styles.ratioFill} ${row.cls}`} style={{ width: pct(row.v / total) }} />
          </div>
          <strong>{pct(row.v)}</strong>
        </div>
      ))}
    </div>
  )
}

export function ReplySection({ report, staggerIndex }: Props) {
  const { me, partner, reply_time: rt, double_text, initiation_ratio } = report
  const max = Math.max(rt.me_box.hi, rt.partner_box.hi, 1)
  const faster = rt.me_median_sec <= rt.partner_median_sec ? me : partner

  return (
    <Section no="03" title="답장 패턴" staggerIndex={staggerIndex}>
      <div className={styles.medians}>
        <div className={styles.medianCard}>
          <p className={styles.medianLabel}>{me}의 답장 속도</p>
          <p className={`${styles.medianValue} ${styles.toneMe}`}>{formatReplyTime(rt.me_median_sec)}</p>
        </div>
        <div className={styles.medianCard}>
          <p className={styles.medianLabel}>{partner}의 답장 속도</p>
          <p className={`${styles.medianValue} ${styles.tonePartner}`}>{formatReplyTime(rt.partner_median_sec)}</p>
        </div>
      </div>

      <div className={styles.boxes}>
        <p className={styles.boxesTitle}>답장 시간 분포 (5–95 퍼센타일, 박스는 IQR)</p>
        <BoxPlot name={me} box={rt.me_box} max={max} color="#1f4d8f" />
        <BoxPlot name={partner} box={rt.partner_box} max={max} color="#c8361f" />
        <p className={styles.insight}>
          <strong>{faster}</strong>님이 대체로 더 빠르고 일관되게 답장합니다.
        </p>
      </div>

      <div className={styles.ratios}>
        <RatioBar label="더블 텍스트 패턴" me={double_text.me} partner={double_text.partner} meName={me} partnerName={partner} />
        <RatioBar label="대화 시작 빈도" me={initiation_ratio} partner={1 - initiation_ratio} meName={me} partnerName={partner} />
      </div>
    </Section>
  )
}
```

```scss
// ReplySection.module.scss
@use '../../styles/tokens' as *;

.medians {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.medianCard {
  background: var(--paper-raised);
  border: 1px solid var(--hairline);
  padding: var(--space-3);
}

.medianLabel {
  @include mono-label;
  margin-bottom: var(--space-1);
}

.medianValue {
  font-family: var(--font-display);
  font-size: 2.6rem;
  font-weight: 900;
  line-height: 1;
}

.toneMe { color: var(--me); }
.tonePartner { color: var(--signal); }

.boxes {
  border: 1px solid var(--hairline);
  background: var(--paper-raised);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
}

.boxesTitle {
  @include mono-label;
  margin-bottom: var(--space-2);
}

.boxRow {
  display: grid;
  grid-template-columns: 5rem 1fr 4rem;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
}

.boxName,
.boxVal {
  font-family: var(--font-mono);
  font-size: 0.8rem;
}

.boxTrack {
  position: relative;
  height: 1.4rem;
}

.whisker {
  position: absolute;
  top: 50%;
  height: 1px;
  background: var(--hairline-strong);
}

.iqr {
  position: absolute;
  top: 0.2rem;
  bottom: 0.2rem;
  border: 1.5px solid;
  background: var(--paper);
}

.median {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
}

.insight {
  font-size: 0.9rem;
  color: var(--ink-soft);
  margin-top: var(--space-1);
}

.ratios {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
}

.ratioCard {
  border: 1px solid var(--hairline);
  background: var(--paper-raised);
  padding: var(--space-3);
}

.ratioTitle {
  font-weight: 600;
  margin-bottom: var(--space-2);
}

.ratioRow {
  display: grid;
  grid-template-columns: 4.5rem 1fr 3.5rem;
  align-items: center;
  gap: var(--space-1);
  margin-bottom: var(--space-1);
  font-family: var(--font-mono);
  font-size: 0.8rem;

  strong {
    text-align: right;
  }
}

.ratioTrack {
  height: 6px;
  background: var(--paper);
  border: 1px solid var(--hairline);
}

.ratioFill {
  display: block;
  height: 100%;
}

.fillMe { background: var(--me); }
.fillPartner { background: var(--signal); }

@media (max-width: 640px) {
  .medians,
  .ratios {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 3: ReportPage에 섹션 추가**

```tsx
<EmotionSection report={report} staggerIndex={4} />
<ReplySection report={report} staggerIndex={5} />
```

(import 추가)

- [ ] **Step 4: 테스트/빌드 + 커밋**

Run: `cd frontend && npm test && npm run build`
Expected: PASS

```bash
git add frontend/src/components/report
git commit -m "feat(frontend): add emotion and reply pattern sections"
```

---

### Task 15: SinceritySection (04) + AISection (05)

**Files:**
- Create: `frontend/src/components/report/SinceritySection.tsx`, `SinceritySection.module.scss`
- Create: `frontend/src/components/report/AISection.tsx`, `AISection.module.scss`
- Modify: `frontend/src/components/report/ReportPage.tsx` (LLM nudge 블록을 AISection으로 대체)

- [ ] **Step 1: SinceritySection.tsx**

```tsx
import type { ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './SinceritySection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function scoreLabel(score: number): { text: string; cls: string } {
  if (score < 0.4) return { text: '낮은 성의도', cls: styles.low }
  if (score > 0.65) return { text: '높은 성의도', cls: styles.high }
  return { text: '보통 성의도', cls: styles.mid }
}

export function SinceritySection({ report, staggerIndex }: Props) {
  const { qa_sincerity: qa } = report

  return (
    <Section no="04" title="성의도" staggerIndex={staggerIndex}>
      <div className={styles.summary}>
        <p className={styles.summaryLabel}>평균 질문-답변 유사도 (SBERT 코사인 유사도)</p>
        <div className={styles.summaryTrack}>
          <span className={styles.summaryFill} style={{ width: pct(qa.avg_sincerity) }} />
        </div>
        <span className={styles.summaryValue}>{pct(qa.avg_sincerity)}</span>
      </div>

      {qa.pairs.length === 0 ? (
        <p className={styles.empty}>분석할 질문-답변 쌍이 충분하지 않습니다.</p>
      ) : (
        <ol className={styles.pairs}>
          {qa.pairs.map((pair, i) => {
            const { text, cls } = scoreLabel(pair.score)
            return (
              <li key={i} className={styles.pair}>
                <div className={styles.bubbleRow}>
                  <div className={styles.bubble}>
                    <span className={styles.speaker}>{pair.questioner}</span>
                    <p>{pair.question}</p>
                  </div>
                  <div className={`${styles.bubble} ${styles.answer}`}>
                    <span className={styles.speaker}>{pair.answerer}</span>
                    <p>{pair.answer}</p>
                  </div>
                </div>
                <div className={styles.scoreRow}>
                  <div className={styles.scoreTrack}>
                    <span className={`${styles.scoreFill} ${cls}`} style={{ width: pct(pair.score) }} />
                  </div>
                  <span className={`${styles.scoreText} ${cls}`}>{pct(pair.score)} · {text}</span>
                </div>
              </li>
            )
          })}
        </ol>
      )}
      <p className={styles.caption}>성의도 낮은 순 최대 10쌍 표시</p>
    </Section>
  )
}
```

```scss
// SinceritySection.module.scss
@use '../../styles/tokens' as *;

.summary {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: var(--space-2);
  border: 1px solid var(--hairline);
  background: var(--paper-raised);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
}

.summaryLabel {
  @include mono-label;
}

.summaryTrack {
  height: 8px;
  background: var(--paper);
  border: 1px solid var(--hairline);
}

.summaryFill {
  display: block;
  height: 100%;
  background: var(--ink);
}

.summaryValue {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 700;
}

.pairs {
  list-style: none;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.pair {
  border: 1px solid var(--hairline);
  background: var(--paper-raised);
  padding: var(--space-2) var(--space-3);
}

.bubbleRow {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-2);
  margin-bottom: var(--space-2);
}

.bubble {
  font-size: 0.9rem;

  p {
    background: var(--me-soft);
    padding: 0.5rem 0.75rem;
    margin-top: 0.25rem;
  }
}

.answer p {
  background: var(--signal-soft);
}

.speaker {
  @include mono-label;
}

.scoreRow {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: var(--space-2);
}

.scoreTrack {
  height: 5px;
  background: var(--paper);
  border: 1px solid var(--hairline);
}

.scoreFill {
  display: block;
  height: 100%;

  &.low { background: var(--signal); }
  &.mid { background: var(--ink-soft); }
  &.high { background: var(--ok); }
}

.scoreText {
  font-family: var(--font-mono);
  font-size: 0.75rem;

  &.low { color: var(--signal); }
  &.mid { color: var(--ink-soft); }
  &.high { color: var(--ok); }
}

.empty {
  color: var(--ink-soft);
}

.caption {
  @include mono-label;
  margin-top: var(--space-2);
}

@media (max-width: 640px) {
  .bubbleRow {
    grid-template-columns: 1fr;
  }

  .summary {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 2: AISection.tsx**

LLM 결과 3상태(null/에러/정상) 처리. 정상 시: 신뢰도 + 축별 비교 테이블 + 근거 인용 + 마크다운 리포트:

```tsx
import ReactMarkdown from 'react-markdown'
import type { AxisComparison, ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './AISection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function ComparisonRow({ axis, comp }: { axis: string; comp: AxisComparison }) {
  return (
    <tr>
      <th scope="row">{axis}</th>
      <td>{comp.tier1.toFixed(2)}</td>
      <td>{comp.llm.toFixed(2)}</td>
      <td className={comp.delta >= 0 ? styles.deltaUp : styles.deltaDown}>
        {comp.delta >= 0 ? '+' : ''}{comp.delta.toFixed(2)}
      </td>
      <td>
        <span className={comp.agree ? styles.agree : styles.disagree}>
          {comp.agree ? '동의' : '관점 차이'}
        </span>
      </td>
    </tr>
  )
}

export function AISection({ report, staggerIndex }: Props) {
  const { llm, llm_error } = report

  return (
    <Section no="05" title="AI 심층 분석" staggerIndex={staggerIndex}>
      {llm === null && llm_error === null && (
        <p className={styles.nudge}>
          OpenAI API 키를 입력하면 LLM이 실제 대화 장면을 인용하며 심층 진단을 제공합니다.
          (규칙 기반 분석 결과는 위에 그대로 유지됩니다)
        </p>
      )}

      {llm_error !== null && (
        <p className={styles.error}>
          LLM 분석을 완료하지 못했어요: {llm_error} (규칙 기반 결과는 정상입니다)
        </p>
      )}

      {llm !== null && (
        <>
          <p className={styles.confidence}>LLM 신뢰도 {pct(llm.confidence)}</p>

          <table className={styles.table}>
            <thead>
              <tr>
                <th>축</th>
                <th>규칙(BERT)</th>
                <th>LLM</th>
                <th>Δ</th>
                <th>일치</th>
              </tr>
            </thead>
            <tbody>
              <ComparisonRow axis="지배성" comp={llm.dominance} />
              <ComparisonRow axis="의존도" comp={llm.dependence} />
            </tbody>
          </table>

          {(['dominance', 'dependence'] as const).map((axis) =>
            (llm.evidence[axis] ?? []).length > 0 && (
              <details key={axis} className={styles.evidence}>
                <summary>{axis === 'dominance' ? '지배성' : '의존도'} 근거 대화 보기</summary>
                {llm.evidence[axis].map((w, i) => (
                  <blockquote key={i} className={styles.quote}>
                    <span className={styles.sim}>유사도 {w.sim.toFixed(2)}</span>
                    <pre>{w.text}</pre>
                  </blockquote>
                ))}
              </details>
            ),
          )}

          <div className={styles.report}>
            <ReactMarkdown>{llm.report}</ReactMarkdown>
          </div>
        </>
      )}
    </Section>
  )
}
```

```scss
// AISection.module.scss
@use '../../styles/tokens' as *;

.nudge {
  color: var(--ink-soft);
  background: var(--paper-raised);
  border: 1px dashed var(--hairline-strong);
  padding: var(--space-3);
}

.error {
  border-left: 3px solid var(--signal);
  background: var(--signal-soft);
  padding: var(--space-2) var(--space-3);
}

.confidence {
  @include mono-label;
  margin-bottom: var(--space-2);
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  margin-bottom: var(--space-3);

  th,
  td {
    border: 1px solid var(--hairline);
    padding: 0.6rem 0.9rem;
    text-align: left;
  }

  thead th {
    @include mono-label;
    background: var(--paper-raised);
  }
}

.deltaUp { color: var(--me); }
.deltaDown { color: var(--signal); }

.agree {
  color: var(--ok);
  font-weight: 600;
}

.disagree {
  color: var(--signal);
  font-weight: 600;
}

.evidence {
  margin-bottom: var(--space-2);

  summary {
    @include mono-label;
    cursor: pointer;
    padding: var(--space-1) 0;
  }
}

.quote {
  margin: var(--space-1) 0;
  padding: var(--space-2);
  background: var(--paper-raised);
  border-left: 2px solid var(--hairline-strong);

  pre {
    font-family: var(--font-body);
    font-size: 0.88rem;
    white-space: pre-wrap;
    margin: 0.25rem 0 0;
  }
}

.sim {
  @include mono-label;
}

.report {
  border-top: 1px solid var(--hairline);
  padding-top: var(--space-3);
  margin-top: var(--space-2);
  // 마크다운 리포트는 세리프 본문으로 — 진단서 느낌
  font-family: var(--font-display);
  line-height: 1.8;

  h2,
  h3 {
    margin: var(--space-2) 0 var(--space-1);
  }

  p,
  li {
    margin-bottom: var(--space-1);
  }
}
```

- [ ] **Step 3: ReportPage 마무리**

기존 LLM nudge `<Section no="—" ...>` 블록을 삭제하고 섹션 추가:

```tsx
<SinceritySection report={report} staggerIndex={6} />
<AISection report={report} staggerIndex={7} />
```

(import 추가. ReportPage.test.tsx의 "API 키를 입력하면" 검증은 이제 AISection이 렌더하므로 여전히 통과해야 함.)

- [ ] **Step 4: 테스트/빌드 확인**

Run: `cd frontend && npm test && npm run build`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/components/report
git commit -m "feat(frontend): add sincerity and AI deep-analysis sections"
```

---

### Task 16: E2E 수동 검증 + README 업데이트

**Files:**
- Modify: `README.md` (실행 방법 추가)

- [ ] **Step 1: 백엔드+프론트 동시 기동 후 전체 플로우 수동 검증**

```bash
# 터미널 1
uv run uvicorn server.main:app --port 8000
# 터미널 2
cd frontend && npm run dev
```

체크리스트:
- [ ] CSV 업로드 → 요약 카드 표시
- [ ] 설정 후 "감사 시작" → 진행 단계가 순서대로 갱신
- [ ] 리포트: 평결 3지표, 레이더, 01~04 섹션 모두 데이터 표시
- [ ] API 키 없이 → AI 섹션에 안내문
- [ ] (키 보유 시) API 키 입력 → AI 섹션에 비교 테이블 + 마크다운 리포트
- [ ] 잘못된 CSV 업로드 → 에러 메시지 표시
- [ ] 모바일 폭(375px)에서 레이아웃 깨짐 없음
- [ ] 전체 백엔드 테스트: `python -m pytest tests/ -v` PASS
- [ ] 전체 프론트 테스트: `cd frontend && npm test` PASS

- [ ] **Step 2: README에 실행 섹션 추가**

기존 README의 실행 방법 부분에 추가:

````markdown
## 웹 앱 실행 (React + FastAPI)

### 백엔드
```bash
uv sync
uv run uvicorn server.main:app --port 8000
```

### 프론트엔드
```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

`/api` 요청은 Vite 프록시를 통해 백엔드(8000)로 전달됩니다.
OpenAI API 키는 업로드 화면에서 선택 입력 — 입력 시 Tier2 LLM 심층 분석이 활성화됩니다.

### 레거시 Streamlit 데모
```bash
uv run streamlit run app.py
```
````

- [ ] **Step 3: 최종 커밋**

```bash
git add README.md
git commit -m "docs: add web app (react + fastapi) run instructions"
```

---

## Self-Review 체크 결과

- **스펙 커버리지:** 업로드+설정(Task 10) / 분석 진행(Task 11) / 결과 리포트: 평결·레이더·대화량·감정·답장·성의도·AI 분석(Task 12–15) / FastAPI 백엔드(Task 2–6) / 에디토리얼 디자인 시스템(Task 7) — Streamlit 화면의 모든 기능이 매핑됨. 사이드바 API 키 입력은 업로드 화면의 선택 입력으로 이전.
- **플레이스홀더 스캔:** 모든 코드 스텝에 실제 코드 포함. "TBD"/"적절히 처리" 류 없음.
- **타입 일관성:** `ReportPayload` 필드명이 Pydantic(Task 2) ↔ serialize(Task 3) ↔ TS(Task 8) ↔ 픽스처(Task 12)에서 동일. `reply_time.me_median_sec` = `partner_to_me_median_sec` 매핑은 Task 3 테스트로 고정. `Section`/`staggerIndex` 시그니처는 Task 12 정의를 13~15에서 동일하게 사용.
- **주의점:** App.tsx 와이어링은 컴포넌트가 생기는 Task 10에서 스텁과 함께 커밋해 빌드 깨짐 방지. 백엔드 테스트는 무거운 HF/SBERT 모델을 주입·monkeypatch로 회피.
