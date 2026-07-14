import { describe, expect, it } from 'vitest'
import { formatReplyTime, interpretIndex, pct } from './format'

describe('formatReplyTime', () => {
  it('formats seconds under a minute', () => {
    expect(formatReplyTime(35)).toBe('35초')
  })
  it('formats minutes under an hour', () => {
    expect(formatReplyTime(150)).toBe('2.5분')
  })
  it('drops trailing zero decimals', () => {
    expect(formatReplyTime(180)).toBe('3분')
  })
  it('formats hours', () => {
    expect(formatReplyTime(5400)).toBe('1.5시간')
  })
})

describe('interpretIndex', () => {
  it('me leads above 0.65', () => {
    expect(interpretIndex(0.7, '지언', '민수')).toEqual({ tone: 'me', text: '지언 쪽이 우위' })
  })
  it('partner leads below 0.35', () => {
    expect(interpretIndex(0.3, '지언', '민수')).toEqual({ tone: 'partner', text: '민수 쪽이 우위' })
  })
  it('balanced in between', () => {
    expect(interpretIndex(0.5, '지언', '민수')).toEqual({ tone: 'balanced', text: '균형' })
  })
})

describe('pct', () => {
  it('formats ratio as percent', () => {
    expect(pct(0.553)).toBe('55%')
    expect(pct(0.553, 1)).toBe('55.3%')
  })
})
