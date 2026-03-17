import pandas as pd
from utils.text_utils import is_system_message, is_non_text, clean_text

SESSION_END_MINUTES = 30
MAX_USERS = 2
BOT = ["플레이봇"]

def parse_kakao_chat(file_path):
  df = pd.read_csv(file_path, header=0, dtype=str, keep_default_na=False, encoding="utf-8-sig")
  
  # 메시지가 삭제된 경우 날짜 공백
  df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
  df = df.dropna(subset=["Date"])
  
  df["Message"] = df["Message"].apply(clean_text)
  df["User"] = df["User"].str.strip()
  
  # 시스템 메시지, 비텍스트 메시지, 봇 메시지, 빈 메시지 제거
  is_valid = (
  ~df["User"].isin(BOT)
  & (df["User"] != "")
  & ~df["Message"].apply(is_system_message)
  & ~df["Message"].apply(is_non_text)
  & (df["Message"] != "")
  )
  df = df[is_valid].reset_index(drop=True)
  
  if df.empty:
    raise ValueError("[Kakao Parser] 분석 가능한 메시지가 없습니다.")

  users = df["User"].unique().tolist()
  if len(users) > MAX_USERS:
    raise ValueError(f"[Kakao Parser] 최대 {MAX_USERS}명까지 지원합니다.")
  elif len(users) < 2:
    raise ValueError("[Kakao Parser] 대화에는 최소 2명의 참여자가 필요합니다.")
  
  return df


def split_sessions(df, session_end_minutes=SESSION_END_MINUTES):
  time_diff = df["Date"].diff()
  is_new_session = time_diff > pd.Timedelta(minutes=session_end_minutes)
  df["Session_ID"] = is_new_session.cumsum()
  return df
  