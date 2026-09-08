import { useEffect, useState } from 'react'
import { classifierRequest } from '../../api'
import PatchOverlay from '../../components/PatchOverlay'

type Explanation = { prediction: string; reason?: string; score_kind?: string; patches: { left: number; top: number; right: number; bottom: number; influence: number }[] }

export default function ClassifierPatchImage({ run, model, index, src, alt, autoExplain = false }: { run: string; model: string; index: number; src: string; alt: string; autoExplain?: boolean }) {
  const [enabled, setEnabled] = useState(false)
  const explain = enabled || autoExplain
  const [result, setResult] = useState<Explanation | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    if (!explain) return
    const controller = new AbortController()
    classifierRequest<Explanation>(`patch_explanation/?run_id=${run}&model_id=${model}&image_index=${index}`, undefined, controller.signal).then(setResult).catch((e: unknown) => { if (!controller.signal.aborted) setError(String(e)) })
    return () => controller.abort()
  }, [explain, run, model, index])
  const positive = result?.prediction === 'good' ? [] : [...(result?.patches ?? [])].filter(p => p.influence > 1e-8).sort((a,b) => b.influence-a.influence).slice(0,5)
  const max = positive[0]?.influence ?? 1
  return <div>
    {!explain && <button onClick={() => setEnabled(true)}>Explain predicted defect patches</button>}
    {explain && !result && !error && <p role="status">Comparing patch contributions…</p>}
    {error && <p role="alert">{error}</p>}
    <PatchOverlay src={src} alt={alt} patches={positive.map(p => ({ ...p, strength: p.influence/max, label: `Removing this patch lowers ${result?.prediction} ${result?.score_kind} by ${p.influence.toPrecision(3)}` }))} description={result ? result.reason ?? (result.prediction === 'good' ? 'Predicted good: no defect patches highlighted.' : positive.length ? `Top ${positive.length} supporting patches for ${result.prediction}. Approximate leave-one-patch-out explanation, not patch-level defect labels. Hover for score changes.` : 'No positive patch contributions found. The prediction may depend on distributed image features.') : 'This classifier predicts the whole image; patch contributions are available on request.'} />
  </div>
}
