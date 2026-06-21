// frontend/src/components/report/VerdictSection.tsx
import type { CSSProperties } from 'react'
import type { ReportPayload } from '../../api/types'
import { interpretIndex, type IndexTone } from '../../lib/format'
import styles from './VerdictSection.module.scss'

const TONE_CLASS: Record<IndexTone, string> = {
  me: styles.toneMe,
  partner: styles.tonePartner,
  balanced: styles.toneBalanced,
}

interface Props {
  report: ReportPayload
}

export function VerdictSection({ report }: Props) {
  const { me, partner, dominance_index, dependence_index, balance } = report
  const items = [
    { label: '지배성 지수', value: dominance_index, note: interpretIndex(dominance_index, me, partner), hint: '1에 가까울수록 내가 대화 주도' },
    { label: '의존도 지수', value: dependence_index, note: interpretIndex(dependence_index, me, partner), hint: '1에 가까울수록 내가 더 의존적' },
    {
      label: '균형 점수', value: balance,
      note: balance >= 0.7
        ? { tone: 'balanced' as const, text: '균형적인 관계' }
        : { tone: 'partner' as const, text: '불균형 감지' },
      hint: '1에 가까울수록 균형',
    },
  ]

  return (
    <div className={styles.verdict} style={{ '--stagger-i': 1 } as CSSProperties}>
      {items.map((item) => (
        <div className={styles.item} key={item.label}>
          <p className={styles.label}>{item.label}</p>
          <p className={`${styles.value} ${TONE_CLASS[item.note.tone]}`}>{item.value.toFixed(2)}</p>
          <p className={`${styles.note} ${TONE_CLASS[item.note.tone]}`}>{item.note.text}</p>
          <p className={styles.hint}>{item.hint}</p>
        </div>
      ))}
    </div>
  )
}
