import { describe, expect, it } from 'vitest'
import { appReducer, initialState, type AppState } from './appState'
import type { ReportPayload, UploadSummary } from '../api/types'

const summary: UploadSummary = {
  upload_id: 'u1', users: ['지언', '민수'], message_count: 10,
  first_date: '2025-01-01', last_date: '2025-06-01',
}

describe('appReducer', () => {
  it('starts at upload phase', () => {
    expect(initialState.phase).toBe('upload')
  })

  it('UPLOADED moves to configure with summary', () => {
    const next = appReducer(initialState, { type: 'UPLOADED', summary })
    expect(next.phase).toBe('configure')
    expect(next.summary).toEqual(summary)
    expect(initialState.phase).toBe('upload') // 원본 불변
  })

  it('ANALYSIS_STARTED moves to analyzing with jobId', () => {
    const configured: AppState = { ...initialState, phase: 'configure', summary }
    const next = appReducer(configured, { type: 'ANALYSIS_STARTED', jobId: 'j1' })
    expect(next.phase).toBe('analyzing')
    expect(next.jobId).toBe('j1')
  })

  it('ANALYSIS_DONE moves to report with payload', () => {
    const report = { me: '지언' } as ReportPayload
    const next = appReducer(
      { ...initialState, phase: 'analyzing', jobId: 'j1' },
      { type: 'ANALYSIS_DONE', report },
    )
    expect(next.phase).toBe('report')
    expect(next.report?.me).toBe('지언')
  })

  it('FAILED keeps summary so user can retry config', () => {
    const next = appReducer(
      { ...initialState, phase: 'analyzing', summary, jobId: 'j1' },
      { type: 'FAILED', error: '기간에 메시지가 없습니다' },
    )
    expect(next.phase).toBe('configure')
    expect(next.error).toContain('기간')
    expect(next.summary).toEqual(summary)
  })

  it('RESET returns to initial state', () => {
    const next = appReducer({ ...initialState, phase: 'report' }, { type: 'RESET' })
    expect(next).toEqual(initialState)
  })
})
