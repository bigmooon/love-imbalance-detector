import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { ReportPayload } from '../../api/types'
import { pct } from '../../lib/format'
import { Section } from './Section'
import { CHART, monoTick } from './chartTheme'
import styles from './VolumeSection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

export function VolumeSection({ report, staggerIndex }: Props) {
  const { me, partner, timeline, participation: p } = report
  const stats = [
    { label: `${me} 메시지 비율`, value: pct(p.message_count_ratio, 1) },
    { label: `${partner} 메시지 비율`, value: pct(1 - p.message_count_ratio, 1) },
    { label: `${me} 평균 길이`, value: `${Math.round(p.avg_length_me)}자` },
    { label: `${partner} 평균 길이`, value: `${Math.round(p.avg_length_partner)}자` },
  ]

  return (
    <Section no="01" title="대화량" staggerIndex={staggerIndex}>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={timeline} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid stroke={CHART.hairline} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="week" tick={monoTick} tickLine={false} axisLine={{ stroke: CHART.hairline }} />
            <YAxis tick={monoTick} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ fontFamily: CHART.mono, fontSize: 12, background: '#fdfbf4', border: `1px solid ${CHART.hairline}` }} />
            <Legend wrapperStyle={{ fontFamily: CHART.mono, fontSize: 12 }} />
            <Line type="monotone" dataKey="me" name={me} stroke={CHART.me} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="partner" name={partner} stroke={CHART.partner} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <dl className={styles.stats}>
        {stats.map((s) => (
          <div key={s.label}>
            <dt>{s.label}</dt>
            <dd>{s.value}</dd>
          </div>
        ))}
      </dl>
    </Section>
  )
}
