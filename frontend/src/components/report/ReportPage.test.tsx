// frontend/src/components/report/ReportPage.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ReportPage } from './ReportPage'
import { fixtureReport } from './fixtureReport'

describe('ReportPage', () => {
  it('renders verdict numbers and names', () => {
    render(<ReportPage report={fixtureReport} onReset={vi.fn()} />)
    expect(screen.getByText('0.62')).toBeTruthy()  // 지배성
    expect(screen.getByText('0.71')).toBeTruthy()  // 의존도
    expect(screen.getAllByText(/지언/).length).toBeGreaterThan(0)
  })

  it('shows llm nudge when llm is null', () => {
    render(<ReportPage report={fixtureReport} onReset={vi.fn()} />)
    expect(screen.getByText(/API 키를 입력하면/)).toBeTruthy()
  })

  it('renders llm comparison when llm payload exists', () => {
    const withLlm = {
      ...fixtureReport,
      llm: {
        confidence: 0.8,
        report: '## AI 진단\n\n관찰 내용',
        dominance: { tier1: 0.62, llm: 0.7, delta: 0.08, agree: true },
        dependence: { tier1: 0.71, llm: 0.5, delta: -0.21, agree: false },
        evidence: { dominance: [{ text: '[나] 보고싶어', sim: 0.72 }], dependence: [] },
      },
    }
    render(<ReportPage report={withLlm} onReset={vi.fn()} />)
    expect(screen.getByText('동의')).toBeTruthy()
    expect(screen.getByText('관점 차이')).toBeTruthy()
    expect(screen.queryByText(/API 키를 입력하면/)).toBeNull()
  })
})
