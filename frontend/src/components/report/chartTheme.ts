// Recharts에 줄 공통 색/스타일. CSS 변수와 동일 값 (SVG 속성엔 변수 사용이 불안정한 곳 대비)
export const CHART = {
  me: '#1f4d8f',
  partner: '#c8361f',
  ink: '#1c1814',
  inkSoft: '#6b6258',
  hairline: 'rgba(28, 24, 20, 0.22)',
  mono: "'IBM Plex Mono', monospace",
} as const

export const monoTick = { fontFamily: CHART.mono, fontSize: 11, fill: CHART.inkSoft }
