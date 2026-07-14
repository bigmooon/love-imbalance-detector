import {
  Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer,
} from 'recharts'
import type { ReportPayload } from '../../api/types'
import { Section } from './Section'
import { CHART, monoTick } from './chartTheme'
import styles from './RadarSection.module.scss'

interface Props {
  report: ReportPayload
  staggerIndex: number
}

export function RadarSection({ report, staggerIndex }: Props) {
  const { radar, me, partner } = report
  const data = radar.categories.map((category, i) => ({
    category,
    me: radar.me[i],
    partner: radar.partner[i],
  }))

  return (
    <Section no="00" title="대화 권력 지도" staggerIndex={staggerIndex}>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={380}>
          <RadarChart data={data} outerRadius="75%">
            <PolarGrid stroke={CHART.hairline} />
            <PolarAngleAxis dataKey="category" tick={monoTick} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
            <Radar name={me} dataKey="me" stroke={CHART.me} fill={CHART.me} fillOpacity={0.22} strokeWidth={2} />
            <Radar name={partner} dataKey="partner" stroke={CHART.partner} fill={CHART.partner} fillOpacity={0.22} strokeWidth={2} />
            <Legend wrapperStyle={{ fontFamily: CHART.mono, fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <p className={styles.caption}>7개 축 모두 0–1 정규화. 바깥쪽일수록 해당 축에서 그 사람의 비중이 큼.</p>
    </Section>
  )
}
