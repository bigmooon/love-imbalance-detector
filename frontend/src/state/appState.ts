import type { ReportPayload, UploadSummary } from '../api/types'

export type Phase = 'upload' | 'configure' | 'analyzing' | 'report'

export interface AppState {
  phase: Phase
  summary: UploadSummary | null
  jobId: string | null
  report: ReportPayload | null
  error: string | null
}

export type AppAction =
  | { type: 'UPLOADED'; summary: UploadSummary }
  | { type: 'ANALYSIS_STARTED'; jobId: string }
  | { type: 'ANALYSIS_DONE'; report: ReportPayload }
  | { type: 'FAILED'; error: string }
  | { type: 'RESET' }

export const initialState: AppState = {
  phase: 'upload',
  summary: null,
  jobId: null,
  report: null,
  error: null,
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'UPLOADED':
      return { ...state, phase: 'configure', summary: action.summary, error: null }
    case 'ANALYSIS_STARTED':
      return { ...state, phase: 'analyzing', jobId: action.jobId, error: null }
    case 'ANALYSIS_DONE':
      return { ...state, phase: 'report', report: action.report, error: null }
    case 'FAILED':
      // 설정 화면으로 돌려보내 재시도 가능하게 (업로드 요약은 보존)
      return { ...state, phase: state.summary ? 'configure' : 'upload', error: action.error }
    case 'RESET':
      return initialState
  }
}
