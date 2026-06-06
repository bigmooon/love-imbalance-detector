// frontend/src/components/report/ReportPage.tsx
import type { CSSProperties } from 'react'
import type { ReportPayload } from '../../api/types'
import { Section } from './Section'
import { VerdictSection } from './VerdictSection'
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

      {/* Task 13~15에서 섹션 추가: RadarSection, VolumeSection, EmotionSection,
          ReplySection, SinceritySection, AISection */}

      {report.llm === null && report.llm_error === null && (
        <Section no="—" title="AI 심층 분석" staggerIndex={6}>
          <p className={styles.llmNudge}>
            OpenAI API 키를 입력하면 LLM이 실제 대화 장면을 인용하며 심층 진단을 제공합니다.
            (규칙 기반 분석 결과는 위에 그대로 유지됩니다)
          </p>
        </Section>
      )}
    </main>
  )
}
