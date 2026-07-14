export function formatReplyTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}초`
  const minutes = seconds / 60
  if (minutes < 60) return `${minutes.toFixed(1).replace(/\.0$/, '')}분`
  const hours = minutes / 60
  return `${hours.toFixed(1).replace(/\.0$/, '')}시간`
}

export type IndexTone = 'me' | 'partner' | 'balanced'

export interface IndexInterpretation {
  tone: IndexTone
  text: string
}

export function interpretIndex(value: number, me: string, partner: string): IndexInterpretation {
  if (value >= 0.65) return { tone: 'me', text: `${me} 쪽이 우위` }
  if (value <= 0.35) return { tone: 'partner', text: `${partner} 쪽이 우위` }
  return { tone: 'balanced', text: '균형' }
}

export function pct(ratio: number, digits = 0): string {
  return `${(ratio * 100).toFixed(digits)}%`
}
