import { useReducer } from 'react'
import { appReducer, initialState } from './state/appState'
import { UploadPage } from './components/UploadPage'
import { AnalyzingPage } from './components/AnalyzingPage'
import { ReportPage } from './components/report/ReportPage'

export default function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)

  if (state.phase === 'analyzing' && state.jobId) {
    return <AnalyzingPage jobId={state.jobId} dispatch={dispatch} />
  }
  if (state.phase === 'report' && state.report) {
    return <ReportPage report={state.report} onReset={() => dispatch({ type: 'RESET' })} />
  }
  return <UploadPage state={state} dispatch={dispatch} />
}
