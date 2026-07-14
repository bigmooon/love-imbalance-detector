export interface UploadSummary {
  upload_id: string
  users: string[]
  message_count: number
  first_date: string
  last_date: string
}

export interface AnalyzeParams {
  upload_id: string
  me: string
  start_date: string
  end_date: string
  session_gap: number
  preset: string
  api_key?: string | null
}

export interface RadarPayload {
  categories: string[]
  me: number[]
  partner: number[]
}

export interface Participation {
  message_count_ratio: number
  char_count_ratio: number
  avg_length_me: number
  avg_length_partner: number
}

export interface TimelinePoint {
  week: string
  me: number
  partner: number
}

export type EmotionGroup = 'joy' | 'anger' | 'sadness' | 'anxiety' | 'hurt' | 'embarrass'

export interface EmotionPayload {
  me: Record<EmotionGroup, number>
  partner: Record<EmotionGroup, number>
  joy_gap: number
  negative_gap: number
}

export interface BoxStats {
  lo: number
  q1: number
  median: number
  q3: number
  hi: number
}

export interface ReplyTimePayload {
  me_median_sec: number
  partner_median_sec: number
  me_box: BoxStats
  partner_box: BoxStats
}

export interface QAPair {
  questioner: string
  question: string
  answerer: string
  answer: string
  score: number
}

export interface QASincerityPayload {
  avg_sincerity: number
  my_sincerity: number
  partner_sincerity: number
  pairs: QAPair[]
}

export interface AxisComparison {
  tier1: number
  llm: number
  delta: number
  agree: boolean
}

export interface EvidenceWindow {
  text: string
  sim: number
}

export interface LLMPayload {
  confidence: number
  report: string
  dominance: AxisComparison
  dependence: AxisComparison
  evidence: Record<string, EvidenceWindow[]>
}

export interface ReportPayload {
  me: string
  partner: string
  dominance_index: number
  dependence_index: number
  balance: number
  radar: RadarPayload
  participation: Participation
  timeline: TimelinePoint[]
  emotion: EmotionPayload
  reply_time: ReplyTimePayload
  double_text: { me: number; partner: number }
  initiation_ratio: number
  qa_sincerity: QASincerityPayload
  llm: LLMPayload | null
  llm_error: string | null
}

export type JobState = 'pending' | 'running' | 'done' | 'error'

export interface JobStatus {
  job_id: string
  status: JobState
  step: number
  total_steps: number
  label: string
  result: ReportPayload | null
  error: string | null
}
