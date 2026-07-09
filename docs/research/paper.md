# 카카오톡 대화 기반 애정 불균형 탐지: 태스크 정의와 2-Tier 하이브리드 시스템의 정량 평가

> **상태**: 골격 (각 섹션은 대응 노트북 실험 완료 시 채워짐)
> **노트북 매핑**: NB01 `notebooks/01_task_and_dataset.ipynb` · NB02 `02_model_selection` · NB03 `03_emoticon_signals` · NB04 `04_llm_judge_reliability` · NB05 `05_end_to_end_eval`

## 초록

<!-- 전체 완성 후 마지막에 작성 -->

## 1. 서론

- 문제 제기: 연인 대화 속 암묵적 권력/애정 불균형은 정성적으로만 논의되어 왔음
- 본 연구의 기여:
  1. **태스크 정의** — Conversational Affection Imbalance Detection을 입력/출력/평가지표를 갖춘 태스크로 공식화 (§3)
  2. **행동 파라미터 기반 합성 벤치마크** — LLM 순환 평가를 피하는 gold label 구축 프로토콜 (§4)
  3. **모델 선정의 정량 근거** — 경량 한국어 모델 vs LLM의 성능·비용·지연 트레이드오프로 2-Tier 설계 정당화 (§6.1)
  4. **비언어 신호의 기여도 정량화** — ㅋㅋ/ㅠㅠ/이모지/이모티콘 피처의 ablation (§6.2)
  5. **LLM 판정 신뢰도 검증** — 재현성·화자 스왑 대칭성·프롬프트 민감도 (§6.3)

## 2. 관련 연구

<!-- NB01 작성 시 채움 -->
- 한국어 감정 분석: KLUE, KOTE, AI Hub 감성대화 말뭉치
- 대화 관계 분석: linguistic accommodation, 대화 권력 분석
- LLM-as-judge와 그 신뢰도 문제

## 3. 태스크 정의 (NB01)

- **입력**: 2인 카카오톡 대화 D = [(화자, 시각, 메시지, 메시지타입), ...]
- **출력**: dominance ∈ [0,1], dependence ∈ [0,1] (0.5 = 균형)
- **평가지표**: MAE, Spearman ρ, sign accuracy(갑/을 방향), 신뢰도 지표(ICC(2,1), 스왑 대칭성 편차)
- gold label 정의: 합성 대화의 생성 행동 파라미터에서 유도 (§4.1)

## 4. 데이터셋 (NB01)

### 4.1 합성 벤치마크
<!-- 생성 프로토콜, 파라미터 → gold 유도식, 규모 통계표 -->

### 4.2 공개 데이터 (감정 분류 벤치마크용)
<!-- KOTE 44라벨 ↔ 60감정 ↔ 6그룹 매핑 (eval/label_maps.py) -->

### 4.3 실사용 데이터 (케이스 스터디)
<!-- 익명화 절차(llm/anonymize.py), 통계만 공개 -->

## 5. 방법론

### 5.1 2-Tier 하이브리드 구조
<!-- Tier1: KLUE-BERT 감정 + KR-SBERT 성의도 + 규칙 피처 / RAG / Tier2: LLM 판정 + 인용 검증 -->

### 5.2 비언어 신호 피처 (NB03)
<!-- laugh_intensity, tears_ratio, emoji_sentiment, emoticon_ratio_gap -->

## 6. 실험

### 6.1 E1: 감정 분류 모델 선정 (NB02)

| 모델 | macro-F1 | 처리량 (msg/s) | 비용 (₩/1만msg) | 지연 |
|---|---|---|---|---|
| hun3359/klue-bert-base-sentiment (현행) | — | — | — | — |
| KcELECTRA 계열 | — | — | — | — |
| gpt-4o-mini (zero-shot) | — | — | — | — |

### 6.2 E2: 비언어 신호 ablation (NB03)

| 피처 구성 | MAE ↓ | Spearman ρ ↑ | sign acc ↑ |
|---|---|---|---|
| 기존 피처만 | — | — | — |
| + 비언어 피처 | — | — | — |

### 6.3 E3: LLM 판정 신뢰도 (NB04)

| 검증 | 지표 | 결과 |
|---|---|---|
| 반복 재현성 (k=10) | ICC(2,1), σ | — |
| 화자 스왑 대칭성 | mean \|s + s' − 1\| | — |
| 근거 순서 셔플 | Δ점수 | — |
| 프롬프트 변형 | Δ점수 | — |

### 6.4 E4: 종단 구성 타당도 (NB05)

| 시스템 | MAE ↓ | Spearman ρ ↑ | sign acc ↑ |
|---|---|---|---|
| Tier1 (규칙+경량모델) | — | — | — |
| Tier2 (LLM 판정) | — | — | — |

## 7. 결과 및 논의

<!-- NB05 후 작성 -->

## 8. 한계

- 합성 데이터 순환성: 행동 파라미터 기반 gold + 생성/판정 모델 분리로 완화했으나, 합성 대화의 자연스러움 한계는 남음
- 실사용 데이터 small-N: 케이스 스터디 수준
- 60감정 → 6그룹 매핑의 임의성

## 참고문헌

<!-- Shrout & Fleiss (1979), KLUE, KOTE, KR-SBERT 등 -->
