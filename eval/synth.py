"""행동 파라미터 기반 합성 카톡 대화 생성기 (script-then-verbalize).

gold label은 LLM의 의견이 아니라 대본(turn script)을 결정하는 행동
파라미터에서 유도된다. LLM은 대본의 각 텍스트 턴을 자연스러운 한국어
문장으로 채우는 역할만 한다 — 생성 모델 ≠ 판정 모델과 함께 순환성을
이중으로 방어한다.

파이프라인: build_spec → build_turn_script(결정적) → verbalize(LLM)
           → assemble(결정적 타임스탬프) → generate_conversation
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from llm.client import call_structured
from eval.datasets import save_conversation

LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
SCENARIOS = ("daily", "date_plan", "argument", "long_distance")
SESSION_COUNT = 10
SPEAKER_A = "A"
SPEAKER_B = "B"

_SCENARIO_DESC = {
  "daily": "일상 잡담(밥, 회사/학교, 피곤함 등)을 나누는 커플",
  "date_plan": "주말 데이트 계획을 잡는 커플",
  "argument": "서운함이 쌓여 가벼운 다툼과 화해가 오가는 커플",
  "long_distance": "장거리 연애 중이라 자주 만나기 어려운 커플",
}


class TurnText(BaseModel):
  index: int
  text: str


class SessionTexts(BaseModel):
  turns: list[TurnText]


def _lerp(lo, hi, t):
  return lo + (hi - lo) * t


def build_spec(l_dom, l_dep, scenario, variant=0):
  """불균형 수준(gold)과 시나리오에서 행동 파라미터 스펙을 만든다.

  l_dom: 1에 가까울수록 A가 갑(선톡·발화량 주도).
  l_dep: 1에 가까울수록 A가 더 의존적(빠른 답장·연속톡·성의·비언어 신호).
  """
  if not (0.0 <= l_dom <= 1.0 and 0.0 <= l_dep <= 1.0):
    raise ValueError(f"불균형 수준은 0~1 범위여야 합니다: dom={l_dom}, dep={l_dep}")
  if scenario not in SCENARIOS:
    raise ValueError(f"알 수 없는 시나리오: {scenario} (지원: {SCENARIOS})")

  behaviors = {
    "initiation_ratio_a": l_dom,
    "message_share_a": _lerp(0.35, 0.65, l_dom),
    "reply_delay_min_a": _lerp(15.0, 1.0, l_dep),
    "reply_delay_min_b": _lerp(1.0, 15.0, l_dep),
    "double_texts_per_session_a": round(_lerp(0.0, 3.0, l_dep)),
    "double_texts_per_session_b": round(_lerp(3.0, 0.0, l_dep)),
    "sincere_answer_a": l_dep >= 0.5,
    "nonverbal_rate_a": _lerp(0.04, 0.30, l_dep),
    "nonverbal_rate_b": _lerp(0.30, 0.04, l_dep),
  }
  cid = (f"dom{int(round(l_dom * 100)):03d}_dep{int(round(l_dep * 100)):03d}"
         f"_{scenario}_v{variant}")
  return {
    "conversation_id": cid,
    "scenario": scenario,
    "variant": variant,
    "gold": {"dominance": l_dom, "dependence": l_dep},
    "behaviors": behaviors,
  }


def _speaker_sequence(rng, starter, n_msgs, share_a, extra_a, extra_b):
  """발화량 목표(share_a)를 정확히 맞추는 화자 순서 생성.

  이후 삽입될 연속톡(extra_a/extra_b — 의존 축 신호)까지 포함한 최종
  비율이 share_a가 되도록 기본 발화 수를 역보정한다. 그렇지 않으면
  dependence 축이 dominance gold(발화량)를 오염시킨다.
  """
  total = n_msgs + extra_a + extra_b
  a_base = round(share_a * total) - extra_a
  a_base = max(1, min(n_msgs - 1, a_base))

  pool = ([SPEAKER_A] * (a_base - (starter == SPEAKER_A))
          + [SPEAKER_B] * ((n_msgs - a_base) - (starter == SPEAKER_B)))
  rng.shuffle(pool)
  return [starter] + pool


def _insert_double_texts(rng, seq, behaviors):
  """의존적 화자의 연속톡을 명시적으로 삽입 (새 리스트 반환)."""
  out = list(seq)
  pairs = (
    (SPEAKER_A, behaviors["double_texts_per_session_a"]),
    (SPEAKER_B, behaviors["double_texts_per_session_b"]),
  )
  for speaker, count in pairs:
    for _ in range(count):
      positions = [i for i, s in enumerate(out) if s == speaker]
      if positions:
        out.insert(rng.choice(positions) + 1, speaker)
  return out


def _base_turn(session, speaker, kind="text"):
  return {
    "session": session, "speaker": speaker, "kind": kind,
    "is_question": False, "answers_question": False,
    "length": "short", "laugh": False, "tears": False,
  }


def _session_turns(rng, session, seq, behaviors):
  """화자 순서에 질문/답변·길이·비언어 신호 지시를 부여한다."""
  turns = []
  prev_question = False
  prev_speaker = None
  for pos, speaker in enumerate(seq):
    answers = prev_question and speaker != prev_speaker
    is_question = (
      not answers and pos < len(seq) - 1
      and seq[pos + 1] != speaker and rng.random() < 0.2
    )
    sincere = (behaviors["sincere_answer_a"] if speaker == SPEAKER_A
               else not behaviors["sincere_answer_a"])
    if answers:
      length = "long" if sincere else "short"
    else:
      length = "medium" if rng.random() < 0.5 else "short"

    turn = {
      **_base_turn(session, speaker),
      "is_question": is_question,
      "answers_question": answers,
      "length": length,
    }
    rate = (behaviors["nonverbal_rate_a"] if speaker == SPEAKER_A
            else behaviors["nonverbal_rate_b"])
    if rng.random() < rate:
      roll = rng.random()
      if roll < 0.5:
        turn = {**turn, "laugh": True}
      elif roll < 0.75:
        turn = {**turn, "tears": True}
      elif not is_question and not answers:
        # 삽입이 아닌 대체: 메시지 수(지배 축 gold)를 오염시키지 않기 위함
        turn = {**_base_turn(session, speaker, kind="emoticon")}
      else:
        turn = {**turn, "laugh": True}  # 질문/답변 턴은 대체 불가 → 웃음으로
    turns.append(turn)
    prev_question = is_question
    prev_speaker = speaker
  return turns


def build_turn_script(spec, seed=0):
  """행동 파라미터를 턴 단위 대본으로 전개한다 (LLM 미사용, 시드 결정적)."""
  rng = random.Random(seed)
  b = spec["behaviors"]

  a_starts = round(b["initiation_ratio_a"] * SESSION_COUNT)
  starters = [SPEAKER_A] * a_starts + [SPEAKER_B] * (SESSION_COUNT - a_starts)
  rng.shuffle(starters)

  turns = []
  for session, starter in enumerate(starters):
    n_msgs = rng.randint(12, 18)
    seq = _speaker_sequence(
      rng, starter, n_msgs, b["message_share_a"],
      b["double_texts_per_session_a"], b["double_texts_per_session_b"],
    )
    seq = _insert_double_texts(rng, seq, b)
    turns.extend(_session_turns(rng, session, seq, b))

  return [{**t, "index": i} for i, t in enumerate(turns)]


def realized_stats(script):
  """대본에서 실제 실현된 행동 지표를 재계산한다 (gold 기록/검증용)."""
  if not script:
    raise ValueError("빈 대본입니다")

  sessions = {}
  for t in script:
    sessions.setdefault(t["session"], []).append(t)
  starters = [turns[0]["speaker"] for turns in sessions.values()]

  a_msgs = [t for t in script if t["speaker"] == SPEAKER_A]
  b_msgs = [t for t in script if t["speaker"] == SPEAKER_B]

  def nonverbal_rate(msgs):
    if not msgs:
      return 0.0
    hits = sum(1 for t in msgs if t["laugh"] or t["tears"] or t["kind"] == "emoticon")
    return hits / len(msgs)

  return {
    "initiation_ratio_a": starters.count(SPEAKER_A) / len(starters),
    "message_share_a": len(a_msgs) / len(script),
    "question_count": sum(1 for t in script if t["is_question"]),
    "nonverbal_rate_a": nonverbal_rate(a_msgs),
    "nonverbal_rate_b": nonverbal_rate(b_msgs),
  }


_SYSTEM_PROMPT = """당신은 한국 커플의 카카오톡 대화를 사실적으로 쓰는 작가입니다.
주어진 대본의 각 턴을 지시에 맞는 자연스러운 한국어 카톡 메시지로 채우세요.

[규칙]
- 반드시 모든 턴(index)에 대해 text를 하나씩 반환하세요.
- length: short=10자 이내 단답, medium=10~30자, long=40자 이상 성의 있는 내용.
- laugh=true면 문장에 ㅋㅋ~ㅋㅋㅋㅋ를, tears=true면 ㅠㅠ를 자연스럽게 포함하세요.
- is_question=true면 상대에게 묻는 문장, answers_question=true면 직전 질문에 대한 답변.
- 실명/지역/연락처 등 개인정보를 만들지 마세요. 화자는 A와 B, 20대 커플입니다.
- 구어체 카톡 말투(축약, 오타 허용)를 사용하세요."""


def _session_prompt(spec, session_turns):
  keys = ("index", "speaker", "is_question", "answers_question", "length", "laugh", "tears")
  lines = [f"시나리오: {_SCENARIO_DESC[spec['scenario']]}", "", "대본:"]
  lines.extend(
    json.dumps({k: t[k] for k in keys}, ensure_ascii=False) for t in session_turns
  )
  return "\n".join(lines)


def verbalize(spec, script, config, client=None):
  """대본의 텍스트 턴을 세션 단위 LLM 호출로 문장화한다.

  Returns:
    {턴 index: 텍스트} 딕셔너리. 누락 턴이 있으면 ValueError.
  """
  texts = {}
  for session in sorted({t["session"] for t in script}):
    session_turns = [t for t in script if t["session"] == session and t["kind"] == "text"]
    result = call_structured(
      config, _SYSTEM_PROMPT, _session_prompt(spec, session_turns),
      SessionTexts, client=client,
    )
    returned = {t.index: t.text for t in result.turns}
    missing = [t["index"] for t in session_turns if t["index"] not in returned]
    if missing:
      raise ValueError(f"세션 {session}에서 채워지지 않은 턴: {missing}")
    texts = {**texts, **{t["index"]: returned[t["index"]] for t in session_turns}}
  return texts


def _decorate(text, turn):
  """LLM이 비언어 지시를 빠뜨린 경우를 대비한 보정."""
  out = text
  if turn["laugh"] and "ㅋ" not in out:
    out = f"{out} ㅋㅋ"
  if turn["tears"] and "ㅠ" not in out:
    out = f"{out} ㅠㅠ"
  return out


def assemble(spec, script, texts, seed=0, base_start="2026-01-05 09:00"):
  """대본+텍스트를 타임스탬프 있는 메시지 목록으로 조립한다 (결정적).

  답장 지연·연속톡 간격·세션 간격(>30분)이 행동 파라미터를 따른다.
  """
  b = spec["behaviors"]
  missing = [t["index"] for t in script if t["kind"] == "text" and t["index"] not in texts]
  if missing:
    raise ValueError(f"텍스트가 없는 턴: {missing[:5]}{'...' if len(missing) > 5 else ''}")

  rng = random.Random(seed + 1)  # 대본용 rng와 분리
  cursor = datetime.strptime(base_start, "%Y-%m-%d %H:%M")
  messages = []
  prev = None
  for t in script:
    if prev is None:
      pass  # 첫 메시지는 base_start 그대로
    elif t["session"] != prev["session"]:
      cursor += timedelta(hours=rng.uniform(4.0, 10.0))
    elif t["speaker"] == prev["speaker"]:
      cursor += timedelta(seconds=rng.uniform(20, 90))
    else:
      delay = b["reply_delay_min_a"] if t["speaker"] == SPEAKER_A else b["reply_delay_min_b"]
      cursor += timedelta(minutes=delay * rng.uniform(0.5, 1.5), seconds=rng.uniform(1, 30))

    text = "이모티콘" if t["kind"] == "emoticon" else _decorate(texts[t["index"]], t)
    messages.append({
      "user": t["speaker"],
      "text": text,
      "datetime": cursor.strftime("%Y-%m-%d %H:%M:%S"),
      "session": t["session"],
      "type": t["kind"],
    })
    prev = t
  return messages


def generate_conversation(spec, config, seed=0, client=None):
  """스펙 하나를 완성된 대화(meta+messages)로 생성한다."""
  script = build_turn_script(spec, seed=seed)
  texts = verbalize(spec, script, config, client=client)
  messages = assemble(spec, script, texts, seed=seed)
  meta = {
    **spec,
    "realized": realized_stats(script),
    "generator_model": config.model,
    "seed": seed,
  }
  return {"meta": meta, "messages": messages}


def generate_dataset(out_dir, config, scenarios=SCENARIOS, levels=LEVELS,
                     variants=5, base_seed=42, client=None, progress=None):
  """벤치마크 전체 생성: len(levels) × len(scenarios) × variants개 대화.

  l_dep은 (수준+시나리오+변형) 인덱스 오프셋으로 결정적으로 배정해
  dominance/dependence 두 축의 조합이 고르게 섞이도록 한다.
  실패한 대화는 건너뛰고 failed에 기록한다 (API 비용이 드는 1회성 배치).
  """
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  written, failed = [], []
  for i, l_dom in enumerate(levels):
    for j, scenario in enumerate(scenarios):
      for v in range(variants):
        l_dep = levels[(i + j + v) % len(levels)]
        spec = build_spec(l_dom, l_dep, scenario, variant=v)
        seed = base_seed + i * 1000 + j * 100 + v
        try:
          convo = generate_conversation(spec, config, seed=seed, client=client)
        except Exception as e:
          failed.append({"conversation_id": spec["conversation_id"], "error": str(e)})
          continue
        path = out / f"{spec['conversation_id']}.json"
        save_conversation(convo, path)
        written.append(str(path))
        if progress:
          progress(len(written), spec["conversation_id"])
  return {"written": written, "failed": failed}
