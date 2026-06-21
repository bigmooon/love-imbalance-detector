import type { ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import styles from './SinceritySection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function scoreLabel(score: number): { text: string; cls: string } {
  if (score < 0.4) return { text: '낮은 성의도', cls: styles.low }
  if (score > 0.65) return { text: '높은 성의도', cls: styles.high }
  return { text: '보통 성의도', cls: styles.mid }
}

export function SinceritySection({ report, staggerIndex }: Props) {
  const { qa_sincerity: qa } = report

  return (
    <Section no="04" title="성의도" staggerIndex={staggerIndex}>
      <div className={styles.summary}>
        <p className={styles.summaryLabel}>평균 질문-답변 유사도 (SBERT 코사인 유사도)</p>
        <div className={styles.summaryTrack}>
          <span className={styles.summaryFill} style={{ width: pct(qa.avg_sincerity) }} />
        </div>
        <span className={styles.summaryValue}>{pct(qa.avg_sincerity)}</span>
      </div>

      {qa.pairs.length === 0 ? (
        <p className={styles.empty}>분석할 질문-답변 쌍이 충분하지 않습니다.</p>
      ) : (
        <ol className={styles.pairs}>
          {qa.pairs.map((pair, i) => {
            const { text, cls } = scoreLabel(pair.score)
            return (
              <li key={i} className={styles.pair}>
                <div className={styles.bubbleRow}>
                  <div className={styles.bubble}>
                    <span className={styles.speaker}>{pair.questioner}</span>
                    <p>{pair.question}</p>
                  </div>
                  <div className={`${styles.bubble} ${styles.answer}`}>
                    <span className={styles.speaker}>{pair.answerer}</span>
                    <p>{pair.answer}</p>
                  </div>
                </div>
                <div className={styles.scoreRow}>
                  <div className={styles.scoreTrack}>
                    <span className={`${styles.scoreFill} ${cls}`} style={{ width: pct(pair.score) }} />
                  </div>
                  <span className={`${styles.scoreText} ${cls}`}>{pct(pair.score)} · {text}</span>
                </div>
              </li>
            )
          })}
        </ol>
      )}
      <p className={styles.caption}>성의도 낮은 순 최대 10쌍 표시</p>
    </Section>
  )
}
