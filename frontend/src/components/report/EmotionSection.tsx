import type { EmotionGroup, ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './EmotionSection.module.scss'

const EMOTIONS: { group: EmotionGroup; label: string; color: string }[] = [
  { group: 'joy', label: '기쁨', color: '#b58900' },
  { group: 'anger', label: '분노', color: '#c8361f' },
  { group: 'sadness', label: '슬픔', color: '#1f4d8f' },
  { group: 'anxiety', label: '불안', color: '#a45a1c' },
  { group: 'hurt', label: '상처', color: '#6b4d8f' },
  { group: 'embarrass', label: '당황', color: '#2a7d6f' },
]

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function StackedBar({ name, dist }: { name: string; dist: Record<EmotionGroup, number> }) {
  return (
    <div className={styles.barRow}>
      <span className={styles.barName}>{name}</span>
      <div className={styles.bar}>
        {EMOTIONS.map(({ group, label, color }) => {
          const ratio = dist[group] ?? 0
          if (ratio <= 0) return null
          return (
            <span
              key={group}
              className={styles.seg}
              style={{ width: `${ratio * 100}%`, background: color }}
              title={`${label} ${pct(ratio)}`}
            />
          )
        })}
      </div>
    </div>
  )
}

export function EmotionSection({ report, staggerIndex }: Props) {
  const { me, partner, emotion } = report
  const warning = emotion.negative_gap > 0.1

  return (
    <Section no="02" title="감정" staggerIndex={staggerIndex}>
      {warning && (
        <p className={styles.warning}>
          <strong>부정 감정 불균형 감지</strong> — {me}님이 부정적인 감정을 더 많이 표현하고 있어요.
          대화에서 더 많은 공감과 이해가 필요할 수 있습니다.
        </p>
      )}

      <div className={styles.bars}>
        <StackedBar name={me} dist={emotion.me} />
        <StackedBar name={partner} dist={emotion.partner} />
      </div>

      <ul className={styles.legend}>
        {EMOTIONS.map(({ group, label, color }) => (
          <li key={group}>
            <span className={styles.dot} style={{ background: color }} />
            {label}
            <span className={styles.legendVals}>
              {pct(emotion.me[group] ?? 0)} / {pct(emotion.partner[group] ?? 0)}
            </span>
          </li>
        ))}
      </ul>
      <p className={styles.caption}>범례 수치: {me} / {partner}</p>
    </Section>
  )
}
