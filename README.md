# 💘 연애 권력 불균형 진단기

> 카카오톡 대화를 AI로 분석해 연인 사이의 우위관계를 수치화하는 Streamlit 웹앱

---

## 프로젝트 개요

"왜 항상 내가 먼저 연락하지?" "왜 나만 빨리 답장하지?"

연인 관계에는 명확하게 드러나지 않는 권련 구조가 존재합니다.
이 프로젝트는 카카오톡 PC 내보내기(`.csv`) 파일을 입력받아, 
HuggingFace 기반 NLP 모델로 대화를 분석하고 두 가지 핵심 지표를 수치화합니다.

---

## 기술 스택

### 언어 & 프레임워크

| 분류 | 기술 | 버전 |
|------|------|------|
| **Language** | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | `>=3.12` |
| **Web Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) | `>=1.35.0` |
| **데이터 처리** | ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) | `>=2.0.0` |
| **데이터 처리** | ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) | `>=1.26.0` |
| **시각화** | ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) | `>=5.20.0` |
| **NLP 모델** | ![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black) | `>=5.3.0` |
| **패키지 관리** | ![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white) | — |

---

## 🔍 사용한 사전학습/파인튜닝 모델 및 라이선스

이 프로젝트는 다음과 같은 사전학습/파인튜닝 언어 모델을 사용합니다.
모델 가중치는 리포지토리에 포함하지 않으며, 실행 시 HuggingFace Hub에서 직접 로드합니다. 

### 1. 감정 분류(Sentiment Classification)
| 항목 | 내용 |
|------|------|
| **모델** | `hun3359/klue-bert-base-sentiment` |
| **Hugging Face** | [hun3359/klue-bert-base-sentiment](https://huggingface.co/hun3359/klue-bert-base-sentiment) |
| **라이선스** | **CC BY-SA 4.0** |
| **용도** | 60개 세부 감정 -> 6개 그룹 분류 -> 화자별 감정 분표 비대칭 측정 | 

** 선정 이유 **
- **한국어 특화**: KLUE-BERT 기반으로 한국어 구어체 텍스트에 강점을 두고 있어 카카오톡 채팅에도 안정적으로 동작을 기대하였습니다.
- **감정 분류 종류**: 감정 분류가 단순 이진분류가 아닌 60가지 감정으로 분류되어 미묘한 연인 간의 관계를 잡아내기 좋을 것이라 기대하였습니다.

- KLUE-BERT 기반으로 파인튜닝된 한국어 감정 분석 모델
- 저작저/출처 표기 및 동일조건 변경허락(ShareAlike) 의무가 있습니다.
- 자세한 사항은 해당 모델의 링크를 참조해주세요.

> 라이선스 전문: https://creativecommons.org/licenses/by-sa/4.0/

### 2. 문장 임베딩 (Sentence Embedding/Sentence Similarity)

| 항목              | 내용                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| **모델**          | `snunlp/KR-SBERT-V40K-klueNLI-augSTS`                                                             |
| **Hugging Face**  | [snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS) |
| **원저작 GitHub** | [snunlp/KR-SBERT](https://github.com/snunlp/KR-SBERT)                                             |
| **라이선스**      | Hugging Face 모델 카드에 별도 명시 없음 — KR-SBERT 공식 GitHub 저자 정책 따름                     |
| **용도**          | 질문-답변 쌍 임베딩 → 코사인 유사도로 성의도 측정|

**선정 이유**
- **NLI 기반 논리적 정합성 반영**: 
  단순 단어 중복이 아닌 질문과 답변이 논리적으로 호응하는지 평가하기에 최적화
  NLI(추론) + STS(유사도) 데이터로 파인튜닝되어 한국어 질문쌍 처리에 특화되어 있습니다.
  > 참고: https://coco0414.tistory.com/92#google_vignette
- **코사인 유사도 기반 지표 설계**: 별도의 처리없이 `코사인 유사도=성의도`로 계산할 수 있어 효율적입니다.

**Citation**
```bibtex
@misc{kr-sbert,
  author = {Park, Suzi and Hyopil Shin},
  title = {KR-SBERT: A Pre-trained Korean-specific Sentence-BERT model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snunlp/KR-SBERT}}
}
```

---

## 아키텍처

```
love-imbalance-detector
├── app.py                  # Streamlit
│
├── features/
│   ├── dominance.py        # 지배성 지표 계산
│   └── dependence.py       # 의존도 지표 계산
│
├── models/
│   ├── hugging_face.py     # 모델 로드 + 배치 추론 (@st.cache_resource)
│   └── emotion_labels.py   # 감정 그룹 매핑
│
├── visualize/
│   └── charts.py           # Plotly 차트 생성
│
└── utils/
    ├── kakao_parser.py     # 카카오톡 CSV 파싱 + 세션 분리
    └── text_utils.py       # 텍스트 전처리, 질문 판별
```

### 흐름

```
CSV 업로드
  → 파싱 + 화자 추출 (kakao_parser)
  → 기간 필터링 + 세션 분리
  → 감정 분류
  → 문장 임베딩 + QA 코사인 유사도
  → 지표 계산 (dominance / dependence)
  → 지수 스코어링 (가중합 → 0~1 정규화)
  → 결과 시각화 (Plotly)
```

---

## 실행 방법

### uv 사용 (권장)

```bash
# 의존성 설치
uv sync

# 앱 실행
streamlit run app.py
```

### pip 사용
```bash
# 의존성 설치
pip instlall -r requirements.txt

# 앱 실행
streamlit run app.py
```

### 카카오톡 CSV 내보내기

1. 카카오톡 PC 앱 실행
2. 분석하고 싶은 1:1 채팅방 진입
3. 우상단 메뉴(채팅방 설정) -> 대화 내용 관리 -> **텍스트 파일로 저장**
4. 저장된 `.csv` 파일을 앱에 업로드

> 본 프로젝트는 현재 **1:1 대화만** 지원합니다.

---

## 주요 기능

### 지배성 지수 계산

6개 지표를 가중합하여 0~1 사이 점수로 정규화합니다.

```python
DEFAULT_WEIGHTS = {
  "initiation_ratio": 0.20, # 대화 시작
  "ending_ratio": 0.10, # 대화 끝
  "message_count_ratio": 0.20, # 메시지 개수
  "char_count_ratio": 0.15, # 글자 수 
  "joy_gap": 0.15, # 긍부정 감정 비율 
  "negative_gap": 0.20, # 부정 감정 비율
}
```

### 의존성 지수 계산
```python
DEFAULT_WEIGHTS = {
  "reply_time_ratio": 0.35, # 답장 시간
  "double_text_ratio": 0.35, # 연속적으로 메시지를 보낸 수 
  "qa_sincerity_gap": 0.30, # 질의응답 성의도 
}
```

### 성의도 (코사인 유사도)

```
sincerity = mean(cosine_similarity(question_vec, answer_vec))
```

질문-답변 쌍을 KR-SBERT로 임베딩한 뒤, 두 벡터의 코사인 유사도를 측정합니다. 유사도가 높을수록 질문에 맞는 답변을 했다는 의미로, 낮으면 "성의 없는 답변"으로 해석합니다.

### 세션 분리

연속 메시지 간 시간 간격이 `session_gap` 분(기본 30분)을 초과하면 새로운 대화 세션으로 분리합니다. 선톡 비율, 대화 종료 비율 계산의 기준이 됩니다.







