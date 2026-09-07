import { useEffect, useState } from 'react'
import { classifierRequest } from './api'
import type { ModelJob } from './api'

type Score = { mean: number; std: number }
type Point = { fraction: number; images: number; min_images: number; max_images: number; failures: string[]; scores?: Record<string, Score> }
type Curve = { points: Point[]; folds: number }
type State = { curve: Curve | null; job: ModelJob | null }

export default function ClassifierLearningCurves({ run, model, canCompute = true }: { run: string; model: string; canCompute?: boolean }) {
  const [data, setData] = useState<State | null>(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    let timer: number
    const poll = async () => {
      try {
        const value = await classifierRequest<State>(`learning_curves/?run_id=${run}&model_id=${model}`, undefined, controller.signal)
        if (controller.signal.aborted) return
        setData(value)
        if (value.job?.error) setError(value.job.error)
        if (value.job?.status === 'queued' || value.job?.status === 'running') timer = window.setTimeout(poll, 1500)
      } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [run, model, refresh])
  const busy = starting || data?.job?.status === 'running' || data?.job?.status === 'queued'
  const generate = async () => {
    setStarting(true); setError('')
    try { await classifierRequest('learning_curves/', { run_id: run, model_id: model }); setRefresh(n => n+1) }
    catch (e) { setError(String(e)) } finally { setStarting(false) }
  }
  const curve = data?.curve
  return <section className="classifier-learning-curves"><h3>Loss / learning curves</h3>
    <p>Training and CV validation error versus training-set size, not optimization epochs. Each point refits a temporary copy using only the training split. Lower is better; a persistent training–validation gap suggests overfitting. Saved classifiers and holdout data are unchanged.</p>
    <button disabled={busy || !canCompute} onClick={() => void generate()}>{busy ? 'Computing curves…' : curve ? 'Recompute learning curves' : 'Generate learning curves for this model'}</button>
    {!canCompute && <p>Saved curves remain viewable. Generating new curves requires a retained fitted model.</p>}
    {busy && <span role="status"> {data?.job?.done ?? 0} / {data?.job?.total || 25} fold fits</span>}
    {error && <p role="alert">{error}</p>}
    {curve && <><p>{curve.folds} CV folds · nested subsets of 20%, 40%, 60%, 80%, and 100% of each fold’s training groups. Green: training. Blue: validation. Whiskers: ±1 standard deviation across folds, not confidence intervals. KNN training error may be zero because each image can match itself.</p>
      <div className="classifier-performance">{['error','balanced_error'].map(metric => {
        const all = curve.points, valid = all.filter(p => p.scores)
        const maxX = Math.max(1, ...all.map(p => p.images))
        const x = (v: number) => 55 + v/maxX*460, y = (v: number) => 235-Math.max(0, Math.min(1,v))*190
        return <div key={metric}><h4>{metric === 'error' ? 'Classification error · 1 − accuracy' : 'Balanced error · 1 − balanced accuracy'}</h4>
          <svg viewBox="0 0 570 300" role="img" aria-label={`${metric} learning curve`} style={{ width: '100%' }}>
            {[0,.25,.5,.75,1].map(v => <g key={v}><line x1="55" x2="515" y1={y(v)} y2={y(v)} stroke="#dce5df"/><text x="48" y={y(v)+4} textAnchor="end" fontSize="12">{v*100}%</text><text x={x(v*maxX)} y="255" textAnchor="middle" fontSize="12">{Math.round(v*maxX)}</text></g>)}
            {['training','validation'].map((split,i) => <g key={split} stroke={i ? '#486bd4' : '#278568'} fill={i ? '#486bd4' : '#278568'}>
              <path d={all.map((p,index) => p.scores ? `${index && all[index-1].scores ? 'L' : 'M'}${x(p.images)},${y(p.scores[`${split}_${metric}`].mean)}` : '').join(' ')} fill="none" strokeWidth="2"/>
              {valid.map(p => { const s=p.scores![`${split}_${metric}`]; return <g key={p.fraction}><line x1={x(p.images)} x2={x(p.images)} y1={y(s.mean-s.std)} y2={y(s.mean+s.std)}/><circle cx={x(p.images)} cy={y(s.mean)} r="4"><title>{split}: {(s.mean*100).toFixed(1)}% ± {(s.std*100).toFixed(1)}%; {p.min_images}–{p.max_images} images/fold</title></circle></g> })}
            </g>)}
            <text x="285" y="284" textAnchor="middle" fontSize="12">Mean training images per fold</text>
          </svg>
        </div>
      })}</div>
      {curve.points.some(p => p.failures.length) && <details><summary>Unavailable curve points</summary>{curve.points.filter(p => p.failures.length).map(p => <p key={p.fraction}>{p.fraction*100}%: {p.failures.join('; ')}</p>)}</details>}
    </>}
  </section>
}
