import { useEffect, useRef, useState, type Dispatch } from 'react'
import { getJob } from '../api/client'
import type { AppAction } from '../state/appState'
import styles from './AnalyzingPage.module.scss'

const POLL_MS = 1500

interface Props {
  jobId: string
  dispatch: Dispatch<AppAction>
}

export function AnalyzingPage({ jobId, dispatch }: Props) {
  const [step, setStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(7)
  const [label, setLabel] = useState('대기 중')
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false

    const poll = async () => {
      try {
        const job = await getJob(jobId)
        if (stopped.current) return
        setStep(job.step)
        setTotalSteps(job.total_steps)
        setLabel(job.label)
        if (job.status === 'done' && job.result) {
          dispatch({ type: 'ANALYSIS_DONE', report: job.result })
          return
        }
        if (job.status === 'error') {
          dispatch({ type: 'FAILED', error: job.error ?? '분석에 실패했습니다.' })
          return
        }
        window.setTimeout(() => void poll(), POLL_MS)
      } catch (e) {
        if (stopped.current) return
        dispatch({ type: 'FAILED', error: e instanceof Error ? e.message : '연결이 끊겼습니다.' })
      }
    }

    void poll()
    return () => {
      stopped.current = true
    }
  }, [jobId, dispatch])

  const progress = totalSteps > 0 ? (step + 1) / totalSteps : 0

  return (
    <main className={styles.page}>
      <p className={styles.kicker}>Audit in progress</p>
      <h1 className={styles.label} key={label}>
        {label}
        <span className={styles.ellipsis} aria-hidden>…</span>
      </h1>
      <div className={styles.track} role="progressbar"
        aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className={styles.fill} style={{ width: `${progress * 100}%` }} />
      </div>
      <p className={styles.counter}>
        {String(step + 1).padStart(2, '0')} / {String(totalSteps).padStart(2, '0')}
      </p>
    </main>
  )
}
