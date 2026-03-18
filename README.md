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
| **ML / 유사도** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | `>=2.0.0` |
| **ML / 유사도** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) | `>=1.4.0` |
| **NLP 모델(Transformers)** | ![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black) | `>=5.3.0` |
| **NLP 모델(Sentence Transformers)** | ![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black) | `>=3.0.0` |
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
- **Transformers 호환**: HuggingFace `pipeline` API와 호환

- KLUE-BERT 기반으로 파인튜닝된 한국어 감정 분석 모델
- 저작저/출처 표기 및 동일조건 변경허락(ShareAlike) 의무가 있습니다.
- 자세한 사항은 해당 모델의 링크를 참조해주세요.

> 라이선스 전문: https://creativecommons.org/licenses/by-sa/4.0/

### 2. 문장 임베딩 (Sentence Embedding)

| 항목              | 내용                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| **모델**          | `snunlp/KR-SBERT-V40K-klueNLI-augSTS`                                                             |
| **Hugging Face**  | [snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS) |
| **원저작 GitHub** | [snunlp/KR-SBERT](https://github.com/snunlp/KR-SBERT)                                             |
| **라이선스**      | Hugging Face 모델 카드에 별도 명시 없음 — KR-SBERT 공식 GitHub 저자 정책 따름                     |
| **용도**          | 질문-답변 쌍 임베딩 → 코사인 유사도로 성의도 측정|

** 선정 이유**
- **NLI 기반 논리적 정합성 반영**  
  단순 단어 중복이 아닌 질문과 답변이 논리적으로 호응하는지 평가하기에 최적화
  NLI(추론) + STS(유사도) 데이터로 파인튜닝되어 프로젝트 목적에 부합
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


