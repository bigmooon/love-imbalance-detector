import type { AnalyzeParams, JobStatus, UploadSummary } from './types'

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `요청 실패 (HTTP ${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // JSON이 아니면 기본 메시지 유지
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export async function uploadChat(file: File): Promise<UploadSummary> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/upload', { method: 'POST', body: form })
  return parseResponse<UploadSummary>(res)
}

export async function startAnalysis(params: AnalyzeParams): Promise<{ job_id: string }> {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  return parseResponse<{ job_id: string }>(res)
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${jobId}`)
  return parseResponse<JobStatus>(res)
}
