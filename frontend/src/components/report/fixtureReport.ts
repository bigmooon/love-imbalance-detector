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
