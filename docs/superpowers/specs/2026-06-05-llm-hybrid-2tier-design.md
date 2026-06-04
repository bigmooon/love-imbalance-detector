# 설계 스펙: 연애 권력 불균형 진단기 → 2-Tier LLM 하이브리드

- **날짜**: 2026-06-05
- **상태**: 승인됨 (구현 계획 대기)
- **목적**: 포트폴리오/취업용. 기존 BERT+규칙 기반 분석에 생성형 LLM 레이어를 더해 "LLM 엔지니어 역량"을 드러낸다.
- **제약**: Claude Code로 약 3시간 내 구현 가능한 범위. 파인튜닝·대규모 학습 제외.

---

## 1. 목표와 비-목표

### 목표
1. 기존 Tier1(BERT 감정분류 + SBERT QA유사도 + 핸드크래프트 가중합)을 **베이스라인으로 유지**한다.
2. **RAG 기반 근거 검색 + LLM-as-judge**로 dominance/dependence를 0~1 동일 척도로 재판단한다.
3. **LLM vs BERT 경량 비교 + 불일치 설명**을 제공한다.
4. **자연어 진단 리포트**(관찰 + 조언)를 생성한다.
5. 비용/스케일 인식 아키텍처를 코드와 UI로 입증한다.

### 비-목표 (YAGNI / 3시간 가드)
- 파인튜닝, 대규모 학습
- 벡터DB (단일 대화 → in-memory 코사인으로 충분)
- 외부 지식베이스 RAG / Wiki (참조 코퍼스 없음 → 과설계)
- 라벨링된 eval 셋, 신뢰도 메트릭 정량화
- LLM 멀티콜 체인 (판단+리포트는 1회 호출로 동시 생성)
- 에이전트 자율 루프 (단, LangGraph verify-refine은 **스트레치 골**로만 둠)

---

## 2. 아키텍처

```
CSV 업로드
  │
  ├─ Tier 1 (기존, 변경 없음) ── 전체 코퍼스, 싸고 빠름 ───────────────┐
  │    parse → sessions → BERT 감정분류 + SBERT QA유사도               │
  │    → 핸드크래프트 피처 → 가중합 → dominance/dependence (0~1)        │  ← 베이스라인
  │    부산물: per-message 감정, QA쌍+성의도, 세션별 통계                │
  │                                                                     │
  ├─ RAG 검색 레이어 (신규) ── SBERT 임베딩으로 축별 근거 retrieve ─────┤  ← 비용 인식 + RAG
  │    · 전체 메시지(윈도우) 임베딩 (기존 SBERT 재사용)                  │
  │    · 축별 쿼리 임베딩 ("일방적 선톡", "성의 없는 답변" 등)           │
  │    · 코사인 top-k 윈도우 retrieve → 토큰 예산 캡                     │
  │                                                                     │
  └─ Tier 2 (신규) ── 검색된 근거 + Tier1 집계 → OpenAI 1회 호출 ───────┘  ← LLM-as-judge
       structured output(Pydantic):
         · dominance/dependence 점수(0~1) + 근거 인용 + rationale
         · 진단 리포트(마크다운) + confidence
                              │
       비교 레이어 (신규): Tier1 vs Tier2 점수 + 델타/일치도 + 불일치 설명
                              │
       시각화: 비교 게이지 + 근거 카드 + 진단 리포트

  [스트레치] LangGraph verify-refine 루프:
       judge → critic(근거가 점수를 뒷받침하나?) → 약하면 re-retrieve 후 재판단
```

### 핵심 설계 원리 (면접 설명 포인트)
- **2-tier 비용 인식**: Tier1(싸고 넓게, 전체 N만 건) → 근거 retrieve → Tier2(비싸고 깊게, 샘플만). "다 때려넣기"의 비용/지연/lost-in-the-middle 회피.
- **RAG 동기의 진정성**: 큐레이션을 의미 검색으로 승격. 기존 SBERT 자산 재사용. 벡터DB는 단일 대화라 불필요 — 의도적 절제.
- **비교 가능성**: LLM이 Tier1과 동일한 0~1 척도로 점수 → baseline 대비 직접 비교.
- **절제된 고급기술**: LangGraph는 분기/루프가 있을 때만 정당. 선형 파이프라인에 억지로 안 씌움 → 스트레치로 분리.

---

## 3. 모듈 구성 (작고 집중된 파일, 불변 패턴)

```
llm/
  __init__.py
  config.py      # OPENAI_API_KEY, 모델명, 토큰예산 등 환경설정 로딩 (python-dotenv)
  client.py      # OpenAI 래퍼: 키 확인, 재시도(백오프), 타임아웃, 에러 → 도메인 예외
  schema.py      # Pydantic 구조화 출력 모델
  retrieval.py   # SBERT 임베딩 기반 축별 근거 검색 (RAG 핵심)
  judge.py       # 프롬프트 빌드 + LLM 호출 + 출력 검증(할루시네이션 필터) → LLMJudgment
  compare.py     # Tier1 vs Tier2 비교, 델타/일치도, 불일치 플래그
  anonymize.py   # LLM 전송 전 화자명 → "나"/"상대" 익명화

llm/graph.py     # [스트레치] LangGraph verify-refine 루프 (핵심과 독립)
```

- 기존 `models/`, `features/`, `utils/`, `visualize/`, `app.py`는 유지. `app.py`에 Tier2 섹션만 추가.
- 모든 함수는 새 객체 반환 (입력 변경 금지). 함수 <50줄, 파일 <800줄.

---

## 4. 데이터 계약

### 입력 원천: 카카오톡 CSV (기존 `utils/kakao_parser.py` 스키마)
파싱 후 DataFrame 컬럼:
- `Date`: datetime (예 `2026-06-05 14:30:00`, `pd.to_datetime`로 파싱)
- `User`: 화자명 문자열 (예 `상대`)
- `Message`: 메시지 본문 (예 `뭐해?`)
- `Session_ID`: int (시간 간격 cumsum 파생, 예 `3`)

### Tier1 → RAG/Tier2 입력 (기존 분석 결과에서 수집)
```python
tier1_result = {
    "dominance_score": float,        # 0~1
    "dependence_score": float,       # 0~1
    "metrics": { ... },              # 기존 raw 피처 (선톡비율, 답장시간 등)
    "emotion_summary": { ... },      # 화자별 감정 그룹 분포
    "qa_pairs": [ {questioner, question, answerer, answer, score}, ... ],
    "messages": DataFrame,           # User, Message, Date, Session_ID
}
```

### RAG 검색 출력
```python
retrieved = {
    "dominance": [ {text, speaker, session_id, sim}, ... ],   # 축별 top-k 윈도우
    "dependence": [ ... ],
}
```
- **쿼리 정의** (축별 자연어 시드, 코드 상수):
  - dominance: "한 사람이 먼저 연락하고 대화를 주도하는 장면", "감정적으로 우위에 있는 발화"
  - dependence: "성의 없이 짧게 답하거나 무시하는 답변", "한쪽이 매달리거나 빠르게 답장하는 장면"
- **윈도우**: 메시지 단건이 아닌 ±N 메시지 묶음(맥락 보존). 기본 N=2.
- **예산 캡**: top-k 합이 토큰 예산(기본 ~120 메시지 또는 tiktoken 카운트)을 넘지 않도록.

### Tier2 구조화 출력 (Pydantic)
```python
class Evidence(BaseModel):
    quote: str        # 검색된 샘플에 실제 존재하는 원문
    speaker: str      # "나" | "상대"
    reason: str

class AxisJudgment(BaseModel):
    score: float                  # 0~1, 0.5=균형 (Tier1과 동일 척도)
    rationale: str
    evidence: list[Evidence]

class LLMJudgment(BaseModel):
    dominance: AxisJudgment
    dependence: AxisJudgment
    report: str         # 마크다운 진단 리포트
    confidence: float   # 0~1
```

### 비교 출력
```python
comparison = {
    "dominance": {"tier1": float, "llm": float, "delta": float, "agree": bool},
    "dependence": {"tier1": float, "llm": float, "delta": float, "agree": bool},
    "agreement_threshold": 0.15,
}
```

---

## 5. LLM 레이어 상세

- **모델**: OpenAI GPT (config로 모델명 지정, 예 `gpt-4o` 계열). Structured Outputs(JSON schema) 사용.
- **프롬프트**: 한국어. 시스템=관계분석 전문가. 입력 = Tier1 집계통계(익명화) + RAG로 검색된 축별 근거 메시지. 지시:
  - 0~1 척도 정의 명시 (기존 Tier1과 동일 방향):
    - dominance: 0=상대 우위(나=을), 0.5=균형, 1=나 우위(나=갑)
    - dependence: 0=상대가 더 의존적, 0.5=균형, 1=나가 더 의존적 (`compute_dependence_index` 정의와 일치)
  - **반드시 제공된 메시지 안에서만 인용** (외부 추측·창작 금지)
  - 근거 부족 시 confidence 낮추도록
- **할루시네이션 방어**: 응답 `evidence.quote`가 실제 검색 샘플에 substring으로 존재하는지 사후 검증 → 없으면 해당 evidence 제거. 전부 제거되면 해당 축 confidence 하향.
- **프라이버시**: `anonymize.py`로 전송 전 화자명을 `나`/`상대`로 치환. UI에 전송 동의 안내 1줄.
- **비용**: 검색으로 입력 토큰을 캡. 1회 호출로 판단+리포트 동시 생성.

---

## 6. UI 변경 (app.py)

기존 결과 화면 하단에 **"🤖 AI 심층 분석"** 섹션 추가:
1. **비교 시각화**: dominance/dependence 각각 Tier1 vs LLM 점수를 그룹 막대 또는 듀얼 게이지로.
2. **일치도 배지**: 델타 < 임계값 → "✅ AI도 동의" / 이상 → "⚠️ 관점 차이" + LLM rationale로 *왜 다른지*.
3. **근거 카드**: 인용 메시지 + speaker + 이유. 축별 정렬.
4. **진단 리포트**: `st.markdown`으로 렌더링.
5. **API 키 입력**: 사이드바 `st.text_input(type="password")` + env 폴백.

### Graceful degradation
- 키 없음 → Tier2 스킵, Tier1 결과는 정상 표시 + 안내 메시지.
- API 오류/타임아웃/레이트리밋 → catch, Tier1 보존, 사용자 친화 메시지, 앱 크래시 없음.

---

## 7. 횡단 관심사

- **에러 처리**: `client.py`에서 인증·네트워크·레이트리밋·스키마 위반을 도메인 예외로 변환. 호출부는 항상 Tier1 폴백 보장.
- **재시도**: 레이트리밋/일시 오류 시 지수 백오프 (max 2~3회).
- **설정/시크릿**: 키는 env(`OPENAI_API_KEY`) 또는 사이드바 입력. 하드코딩 금지. `.env`는 gitignore.
- **의존성 추가**: `openai`, `pydantic`(설치됨), `python-dotenv`. 선택 `tiktoken`. 스트레치 시 `langgraph`.
- **불변성**: 모든 변환은 새 객체 반환. DataFrame in-place 변경 금지.

---

## 8. 테스트 전략

pytest 기준, 실제 API 호출 없이 **OpenAI mock**:
- `retrieval`: 고정 입력 → 결정적 top-k 선택 검증 (임베딩은 stub/소형 벡터).
- `schema`: 유효/무효 LLM 출력 파싱 및 검증.
- `judge`: mock 응답 → 할루시네이션 필터(존재하지 않는 인용 제거) 동작 검증.
- `compare`: 델타/일치도 경계값 검증.
- `anonymize`: 화자명 치환 정확성.
- `client`: 키 없음/에러 경로의 폴백 동작 (mock).

순수 로직(retrieval/compare/schema/anonymize) 우선 커버. 3시간 예산 내 핵심 경로 중심.

---

## 9. 구현 순서 (우선순위 / 3시간 배분 가이드)

1. **(P0)** `config.py` + `client.py` + `schema.py` — OpenAI 래퍼, 환경설정, 스키마. (~30분)
2. **(P0)** `anonymize.py` + `retrieval.py` — 익명화 + RAG 근거 검색 (SBERT 재사용). (~40분)
3. **(P0)** `judge.py` — 프롬프트 + 호출 + 할루시네이션 필터 → LLMJudgment. (~40분)
4. **(P0)** `compare.py` — Tier1 vs Tier2 비교. (~15분)
5. **(P0)** `app.py` UI 섹션 + graceful degradation. (~35분)
6. **(P1)** 핵심 모듈 단위 테스트 (mock). (~20분)
7. **(스트레치)** `llm/graph.py` — LangGraph verify-refine 루프. (시간 남으면)

P0이 핵심 데모를 완성한다. 시간 부족 시 테스트 범위를 축소하고 스트레치는 생략한다.

---

## 10. 리스크와 완화

| 리스크 | 완화 |
|--------|------|
| 3시간 초과 | P0 우선, 스트레치(LangGraph) 분리, 테스트 범위 가변 |
| 카톡 데이터 프라이버시 | 전송 전 익명화 + UI 동의 안내 |
| LLM 인용 할루시네이션 | 사후 substring 검증으로 필터 |
| 한국어 구조화 출력 품질 | Structured Outputs(스키마 강제) + 프롬프트에 척도/제약 명시 |
| API 키 부재로 데모 불가 | Tier1 단독 동작 보장 (graceful degradation) |
| 토큰 비용 폭증 | RAG 검색으로 입력 캡, 1회 호출 |
