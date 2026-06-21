import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AnalyzingPage } from './AnalyzingPage'
import type { JobStatus } from '../api/types'

afterEach(() => vi.restoreAllMocks())

function jobStatus(overrides: Partial<JobStatus>): JobStatus {
  return {
    job_id: 'j1', status: 'running', step: 2, total_steps: 7,
    label: '감정 분류 중', result: null, error: null, ...overrides,
  }
}

describe('AnalyzingPage', () => {
  it('shows current step label from job status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(jobStatus({}))),
    ))
    render(<AnalyzingPage jobId="j1" dispatch={vi.fn()} />)
    await waitFor(() => expect(screen.getByText(/감정 분류 중/)).toBeTruthy())
    expect(screen.getByText('03 / 07')).toBeTruthy()
  })

  it('dispatches FAILED when job errors', async () => {
    const dispatch = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(jobStatus({ status: 'error', error: '기간 오류' }))),
    ))
    render(<AnalyzingPage jobId="j1" dispatch={dispatch} />)
    await waitFor(() =>
      expect(dispatch).toHaveBeenCalledWith({ type: 'FAILED', error: '기간 오류' }),
    )
  })

  it('dispatches FAILED when done without result', async () => {
    const dispatch = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(jobStatus({ status: 'done', result: null }))),
    ))
    render(<AnalyzingPage jobId="j1" dispatch={dispatch} />)
    await waitFor(() =>
      expect(dispatch).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'FAILED' }),
      ),
    )
  })
})
