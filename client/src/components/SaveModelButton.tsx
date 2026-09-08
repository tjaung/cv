import { useState } from 'react'
import { saveResultModel } from '../api'

export default function SaveModelButton({ family, modelId, runId, saved, available }: { family: 'anomaly' | 'classifiers'; modelId: string; runId?: string; saved?: boolean; available?: boolean }) {
  const [complete, setComplete] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  return <div><button disabled={busy || saved || complete || available === false} onClick={async () => {
    setBusy(true); setError('')
    try { await saveResultModel(family, modelId, runId); setComplete(true) } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }}>{saved || complete ? 'Model saved' : busy ? 'Saving…' : available === false ? 'Results only' : 'Save model'}</button>{error && <p role="alert">{error}</p>}</div>
}
