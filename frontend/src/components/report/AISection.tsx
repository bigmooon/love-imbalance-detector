import ReactMarkdown from 'react-markdown'
import type { AxisComparison, ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './AISection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function ComparisonRow({ axis, comp }: { axis: string; comp: AxisComparison }) {
  return (
    <tr>
      <th scope="row">{axis}</th>
      <td>{comp.tier1.toFixed(2)}</td>
      <td>{comp.llm.toFixed(2)}</td>
      <td className={comp.delta >= 0 ? styles.deltaUp : styles.deltaDown}>
        {comp.delta >= 0 ? '+' : ''}{comp.delta.toFixed(2)}
      </td>
      <td>
        <span className={comp.agree ? styles.agree : styles.disagree}>
          {comp.agree ? '동의' : '관점 차이'}
        </span>
      </td>
    </tr>
  )
}

export function AISection({ report, staggerIndex }: Props) {
  const { llm, llm_error } = report

  return (
    <Section no="05" title="AI 심층 분석" staggerIndex={staggerIndex}>
      {llm === null && llm_error === null && (
        <p className={styles.nudge}>
          OpenAI API 키를 입력하면 LLM이 실제 대화 장면을 인용하며 심층 진단을 제공합니다.
          (규칙 기반 분석 결과는 위에 그대로 유지됩니다)
        </p>
      )}

      {llm_error !== null && (
        <p className={styles.error}>
          LLM 분석을 완료하지 못했어요: {llm_error} (규칙 기반 결과는 정상입니다)
        </p>
      )}

      {llm !== null && (
        <>
          <p className={styles.confidence}>LLM 신뢰도 {pct(llm.confidence)}</p>

          <table className={styles.table}>
            <thead>
              <tr>
                <th>축</th>
                <th>규칙(BERT)</th>
                <th>LLM</th>
                <th>Δ</th>
                <th>일치</th>
              </tr>
            </thead>
            <tbody>
              <ComparisonRow axis="지배성" comp={llm.dominance} />
              <ComparisonRow axis="의존도" comp={llm.dependence} />
            </tbody>
          </table>

          {(['dominance', 'dependence'] as const).map((axis) =>
            (llm.evidence[axis] ?? []).length > 0 && (
              <details key={axis} className={styles.evidence}>
                <summary>{axis === 'dominance' ? '지배성' : '의존도'} 근거 대화 보기</summary>
                {llm.evidence[axis].map((w, i) => (
                  <blockquote key={i} className={styles.quote}>
                    <span className={styles.sim}>유사도 {w.sim.toFixed(2)}</span>
                    <pre>{w.text}</pre>
                  </blockquote>
                ))}
              </details>
            ),
          )}

          <div className={styles.report}>
            <ReactMarkdown>{llm.report}</ReactMarkdown>
          </div>
        </>
      )}
    </Section>
  )
}
