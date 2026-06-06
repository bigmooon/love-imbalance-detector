import type { BoxStats, ReportPayload } from '../../api/types'
import { formatReplyTime, pct } from '../../lib/format'
import { Section } from './Section'
import styles from './ReplySection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

function BoxPlot({ name, box, max, color }: { name: string; box: BoxStats; max: number; color: string }) {
  const x = (v: number) => `${(v / max) * 100}%`
  const w = (a: number, b: number) => `${((b - a) / max) * 100}%`
  return (
    <div className={styles.boxRow}>
      <span className={styles.boxName}>{name}</span>
      <div className={styles.boxTrack}>
        <span className={styles.whisker} style={{ left: x(box.lo), width: w(box.lo, box.hi) }} />
        <span className={styles.iqr} style={{ left: x(box.q1), width: w(box.q1, box.q3), borderColor: color }} />
        <span className={styles.median} style={{ left: x(box.median), background: color }} />
      </div>
      <span className={styles.boxVal}>{box.median.toFixed(1)}분</span>
    </div>
  )
}

function RatioBar({ label, me, partner, meName, partnerName }: {
  label: string; me: number; partner: number; meName: string; partnerName: string
}) {
  const total = me + partner || 1
  return (
    <div className={styles.ratioCard}>
      <p className={styles.ratioTitle}>{label}</p>
      {[{ name: meName, v: me, cls: styles.fillMe }, { name: partnerName, v: partner, cls: styles.fillPartner }].map((row) => (
        <div key={row.name} className={styles.ratioRow}>
          <span>{row.name}</span>
          <div className={styles.ratioTrack}>
            <span className={`${styles.ratioFill} ${row.cls}`} style={{ width: pct(row.v / total) }} />
          </div>
          <strong>{pct(row.v)}</strong>
        </div>
      ))}
    </div>
  )
}

export function ReplySection({ report, staggerIndex }: Props) {
  const { me, partner, reply_time: rt, double_text, initiation_ratio } = report
  const max = Math.max(rt.me_box.hi, rt.partner_box.hi, 1)
  const faster = rt.me_median_sec <= rt.partner_median_sec ? me : partner

  return (
    <Section no="03" title="답장 패턴" staggerIndex={staggerIndex}>
      <div className={styles.medians}>
        <div className={styles.medianCard}>
          <p className={styles.medianLabel}>{me}의 답장 속도</p>
          <p className={`${styles.medianValue} ${styles.toneMe}`}>{formatReplyTime(rt.me_median_sec)}</p>
        </div>
        <div className={styles.medianCard}>
          <p className={styles.medianLabel}>{partner}의 답장 속도</p>
          <p className={`${styles.medianValue} ${styles.tonePartner}`}>{formatReplyTime(rt.partner_median_sec)}</p>
        </div>
      </div>

      <div className={styles.boxes}>
        <p className={styles.boxesTitle}>답장 시간 분포 (5–95 퍼센타일, 박스는 IQR)</p>
        <BoxPlot name={me} box={rt.me_box} max={max} color="#1f4d8f" />
        <BoxPlot name={partner} box={rt.partner_box} max={max} color="#c8361f" />
        <p className={styles.insight}>
          <strong>{faster}</strong>님이 대체로 더 빠르고 일관되게 답장합니다.
        </p>
      </div>

      <div className={styles.ratios}>
        <RatioBar label="더블 텍스트 패턴" me={double_text.me} partner={double_text.partner} meName={me} partnerName={partner} />
        <RatioBar label="대화 시작 빈도" me={initiation_ratio} partner={1 - initiation_ratio} meName={me} partnerName={partner} />
      </div>
    </Section>
  )
}
