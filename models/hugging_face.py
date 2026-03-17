import numpy as np
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from models.emotion_labels import LABEL2ID, get_emotion_group, ACTIVE_PRESET

BATCH_SIZE = 64
SBERT_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"


@st.cache_resource
def load_emotion_classifier():
  """
  HuggingFace 감정 분류 파이프라인 로드.
  @st.cache_resource로 캐싱하여 Streamlit 재실행 시 재로드 방지.
  """
  return pipeline(
    "text-classification",
    model=ACTIVE_PRESET,
    top_k=1,
  )


@st.cache_resource
def load_sbert_model():
  """
  KR-SBERT 문장 임베딩 모델 로드.
  @st.cache_resource로 캐싱하여 Streamlit 재실행 시 재로드 방지.
  """
  return SentenceTransformer(SBERT_MODEL_NAME)

def classify_emotions(texts, classifier):
  """
  텍스트 리스트에 감정 라벨, 그룹, 점수를 배치 단위로 변환
  """
  if not texts:
    return []
  
  # 리스트 초기화
  results = [None] * len(texts)
  
  for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]
    batch_out = classifier(batch)
    
    for j, raw in enumerate(batch_out):
      top = raw[0] if isinstance(raw, list) else raw
      label = top["label"]
      results[start + j] = {
        "label": label,
        "label_id": LABEL2ID.get(label, -1),
        "group": get_emotion_group(label),
        "score": top["score"],
      }
  
  return results

def encode_sentences(texts, model):
  """
  텍스트 리스트를 SBERT 임베딩 벡터로 변환.

  Returns:
    (len(texts), embedding_dim) shape의 numpy 배열.
  """
  if not texts:
    return np.array([])

  return model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=False,
    convert_to_numpy=True,
  )