import { useRef, useState, type Dispatch } from 'react'
import { startAnalysis, uploadChat } from '../api/client'
import type { AppAction, AppState } from '../state/appState'
import styles from './UploadPage.module.scss'

const PRESETS = ['기본', '답장속도 중시', '감정 중시']

interface Props {
  state: AppState
  dispatch: Dispatch<AppAction>
}

export function UploadPage({ state, dispatch }: Props) {
  const { summary, error } = state
  const fileInput = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [me, setMe] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [preset, setPreset] = useState(PRESETS[0])
  const [sessionGap, setSessionGap] = useState(30)
  const [apiKey, setApiKey] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)

  const handleFile = async (file: File) => {
    setBusy(true)
    setLocalError(null)
    try {
      const uploaded = await uploadChat(file)
      setMe(uploaded.users[0])
      setStartDate(uploaded.first_date)
      setEndDate(uploaded.last_date)
      dispatch({ type: 'UPLOADED', summary: uploaded })
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '업로드에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  const handleAnalyze = async () => {
    if (!summary) return
    if (startDate > endDate) {
      setLocalError('시작일이 종료일보다 늦습니다.')
      return
    }
    setBusy(true)
    setLocalError(null)
    try {
      const { job_id } = await startAnalysis({
        upload_id: summary.upload_id,
        me,
        start_date: startDate,
        end_date: endDate,
        session_gap: sessionGap,
        preset,
        api_key: apiKey || null,
      })
      dispatch({ type: 'ANALYSIS_STARTED', jobId: job_id })
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : '분석 시작에 실패했습니다.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.masthead}>
        <p className={styles.kicker}>Relationship Audit</p>
        <h1 className={styles.title}>
          당신과 그 사람,
          <br />
          누가 더 <em>기울어져</em> 있나요?
        </h1>
        <p className={styles.lede}>
          카카오톡 대화를 AI가 읽고, 관계의 권력 불균형을 진단합니다.
          데이터는 분석에만 쓰이고 저장되지 않습니다.
        </p>
      </header>

      {!summary && (
        <section
          className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            if (busy) return
            const file = e.dataTransfer.files[0]
            if (file) void handleFile(file)
          }}
          onClick={() => !busy && fileInput.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              fileInput.current?.click()
            }
          }}
        >
          <input
            ref={fileInput}
            type="file"
            accept=".csv"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void handleFile(file)
            }}
          />
          <p className={styles.dropLabel}>{busy ? '읽는 중…' : 'CSV 파일을 끌어다 놓거나 클릭'}</p>
          <p className={styles.dropHint}>카카오톡 PC → 대화방 메뉴 → 대화 내보내기 → CSV</p>
        </section>
      )}

      {summary && (
        <section className={styles.configure}>
          <dl className={styles.summaryRow}>
            <div>
              <dt>총 메시지</dt>
              <dd>{summary.message_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>첫 대화</dt>
              <dd>{summary.first_date}</dd>
            </div>
            <div>
              <dt>마지막 대화</dt>
              <dd>{summary.last_date}</dd>
            </div>
          </dl>

          <div className={styles.formGrid}>
            <label>
              <span>나는 누구인가요</span>
              <select value={me} onChange={(e) => setMe(e.target.value)}>
                {summary.users.map((u) => (
                  <option key={u} value={u}>{u}</option>
                ))}
              </select>
            </label>
            <label>
              <span>가중치 프리셋</span>
              <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                {PRESETS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </label>
            <label>
              <span>분석 시작일</span>
              <input type="date" value={startDate} min={summary.first_date} max={summary.last_date}
                onChange={(e) => setStartDate(e.target.value)} />
            </label>
            <label>
              <span>분석 종료일</span>
              <input type="date" value={endDate} min={summary.first_date} max={summary.last_date}
                onChange={(e) => setEndDate(e.target.value)} />
            </label>
            <label>
              <span>세션 구분 간격 — {sessionGap}분</span>
              <input type="range" min={10} max={120} step={5} value={sessionGap}
                onChange={(e) => setSessionGap(Number(e.target.value))} />
            </label>
            <label>
              <span>OpenAI API 키 (선택 — AI 심층 분석)</span>
              <input type="password" value={apiKey} placeholder="sk-…"
                onChange={(e) => setApiKey(e.target.value)} autoComplete="off" />
            </label>
          </div>

          <button className={styles.cta} onClick={() => void handleAnalyze()} disabled={busy}>
            {busy ? '시작 중…' : '감사 시작 →'}
          </button>
          <button className={styles.resetLink} onClick={() => dispatch({ type: 'RESET' })}>
            다른 파일 업로드
          </button>
        </section>
      )}

      {(localError ?? error) && <p className={styles.error}>{localError ?? error}</p>}
    </main>
  )
}
