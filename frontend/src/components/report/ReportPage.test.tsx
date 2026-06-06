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
})
