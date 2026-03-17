import re

SYSTEM_MESSAGE_KEYWORDS = [
  "님이 들어왔습니다",
  "님이 나갔습니다",
  "님을 초대했습니다",
  "메시지가 삭제되었습니다."
  
  "원을 받았어요.",
  "원을 보냈어요.",
  "정산 내용을 확인해주세요.",
  "송금봉투가 도착했어요",
  "송금봉투를 받았어요",
  "자동환불 예정",
  "자동환불 완료",
]

# TODO: 이모티콘은 추후 앞뒤 맥락 파악 후 감정 분석에 활용할 수 있도록
NONE_TEXT_PLACEHOLDERS = ["사진", "동영상", "이모티콘", "음성 메시지", "파일", "지도", "연락처", "일정", "투표"]

_QUESTION_ENDINGS = re.compile(
  r"("
  r"[가나니까랴려냐는걸건데할임감슴습]\?|"  # 어미 + 물음표
  r"\?\s*$|"                                  # 단순 물음표 종결
  r"[가나니까랴려냐]\s*$|"                     # 물음표 없는 의문형 어미
  r"[뭐뭘무엇왜어디누구언제어떻얼마몇].*$"  # 의문사가 포함된 문장
  r")"
)

def is_system_message(text):
  return any(keyword in text for keyword in SYSTEM_MESSAGE_KEYWORDS)


def is_non_text(text):
  """
  메시지 중 비텍스트 요소 제거
  - 리스트 단어 중 하나로 시작
  - 뒤에 [공백 + 숫자 + 장/개]가 올 수 있음
  - 문장이 종료됨
  """
  stripped = text.strip()
  pattern = r"^(" + "|".join(NONE_TEXT_PLACEHOLDERS) + r")(\s*\d+[장개]?)?$"
    
  return bool(re.match(pattern, stripped))


def clean_text(text):
  """
  텍스트에서 불필요한 공백과 줄바꿈 제거
  """
  if not isinstance(text, str):
    return ""
  return text.strip()
  
def is_question(text):
  """
  텍스트가 질문인지 여부 판단
  - 물음표로 끝나거나 의문형 어미로 끝나는 경우
  - 의문사가 포함된 경우
  """
  if not text:
    return False
 
  stripped = text.strip()
  if "?" in stripped:
    return True
 
  return bool(_QUESTION_ENDINGS.search(stripped))