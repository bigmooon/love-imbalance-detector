# 데이터 카드 — 애정 불균형 탐지 벤치마크

## 구성

| 디렉토리 | 내용 | 커밋 여부 | 라이선스 |
|---|---|---|---|
| `synth/` | LLM 합성 커플 카톡 대화 + gold label | ✅ 커밋 | 프로젝트 라이선스 (합성물, 개인정보 없음) |
| `public/` | KOTE 등 공개 코퍼스 캐시 | ❌ gitignore | 원 데이터셋 라이선스 준수, 재배포 금지 |
| `private/` | 실사용 카톡 데이터 (케이스 스터디) | ❌ gitignore | 비공개 — 논문에는 익명화된 통계만 인용 |

## 합성 벤치마크 (`synth/`)

- **생성**: `eval/synth.py` — script-then-verbalize 방식.
  행동 파라미터(선톡 비율, 답장 지연, 발화량, 연속톡, 성의, 비언어 신호 빈도)가
  결정적 대본을 만들고, LLM은 각 턴의 한국어 문장화만 담당한다.
- **gold label**: LLM 의견이 아니라 대본을 만든 파라미터에서 유도
  (`meta.gold` = 목표값, `meta.realized` = 대본에서 재계산한 실현값).
- **규모**: 불균형 수준 5단계 × 시나리오 4종 × 변형 5개 = 100개 대화 (각 10세션, 120~180메시지).
- **파일 형식**: 대화당 JSON 1개 —
  `meta{conversation_id, scenario, gold, behaviors, realized, generator_model, seed}` +
  `messages[{user, text, datetime("%Y-%m-%d %H:%M:%S"), session, type}]`.
- **재현**: 같은 seed + 같은 생성 모델이면 대본은 완전 동일, 문장은 모델 출력에 따라 달라질 수 있음.

## 공개 데이터 (`public/`)

- **KOTE** (`searle-j/kote`, HF Hub): 온라인 댓글 5만 건, 44개 감정 다중 라벨.
  감정 분류 벤치마크(NB02)용. 44라벨 → 6그룹 매핑은 `eval/label_maps.py` (저자 판단 기반).
- **AI Hub 감성대화 말뭉치**: 승인 후 추가 예정 (병행 신청, 블로킹 없음).

## 한계

- 합성 대화의 자연스러움은 생성 LLM 능력에 의존하며, 실제 커플 대화의 분포와 다를 수 있다.
- 생성 모델(gpt-4o) ≠ 판정 모델(gpt-4o-mini)로 분리했지만 같은 계열 모델이라는 한계가 있다.
- KOTE는 커플 대화가 아닌 온라인 댓글 도메인이다 (도메인 차이는 논문 §8에 기술).
