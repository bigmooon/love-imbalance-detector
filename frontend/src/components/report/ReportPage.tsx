// frontend/src/components/report/ReportPage.tsx
import type { CSSProperties } from 'react'
import type { ReportPayload } from '../../api/types'
import { VerdictSection } from './VerdictSection'
import { RadarSection } from './RadarSection'
import { VolumeSection } from './VolumeSection'
import { EmotionSection } from './EmotionSection'
import { ReplySection } from './ReplySection'
import { SinceritySection } from './SinceritySection'
import { AISection } from './AISection'
import styles from './ReportPage.module.scss'

interface Props {
  report: ReportPayload
  onReset: () => void
}

export function ReportPage({ report, onReset }: Props) {
  const { me, partner } = report

  return (
    <main className={styles.page}>
      <header className={styles.masthead} style={{ '--stagger-i': 0 } as CSSProperties}>
        <div className={styles.mastRow}>
          <p className={styles.kicker}>Relationship Audit — Final Report</p>
          <button className={styles.reset} onClick={onReset}>새 분석 ↺</button>
        </div>
        <h1 className={styles.title}>
          {me} <span className={styles.vs}>&times;</span> {partner}
        </h1>
        <p className={styles.sub}>대화 권력 불균형 진단 결과</p>
      </header>

      <VerdictSection report={report} />

      <RadarSection report={report} staggerIndex={2} />
      <VolumeSection report={report} staggerIndex={3} />

      <EmotionSection report={report} staggerIndex={4} />
      <ReplySection report={report} staggerIndex={5} />

      <SinceritySection report={report} staggerIndex={6} />
      <AISection report={report} staggerIndex={7} />
    </main>
  )
}
