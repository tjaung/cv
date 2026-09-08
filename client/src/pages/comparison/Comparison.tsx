import ComparisonWriteup from './ComparisonWriteup'
import { useEffect, useRef, useState } from 'react'
import { comparisonRequest, comparisonImageUrl, getModelJob } from '../../api'
import type { ModelJob } from '../../api'

type SavedMetrics = { images: number; binary_accuracy: number | null; class_accuracy: number | null; defect_recall: number | null; false_positive_rate: number | null; missed_defects: number; macro_f1: number | null }
type SavedModel = { loaded?: boolean; metrics?: SavedMetrics | null; id: string; family: string; name: string; available: boolean; reason: string | null; config: Record<string, unknown>; warmup_succeeded?: boolean }
type Sample = { id: number; path: string; actual: string; original_split: string; transformed: boolean; rotation_degrees: number }
type Prediction = { model_id: string; sample_id: number; actual: string; prediction?: string; elapsed_ms: number | null; error?: string; correct?: boolean; supports_classes?: boolean; exposure?: string; score?: number; score_kind?: string }
type Metrics = { total: number; evaluated: number; errors: number; tp: number; tn: number; fp: number; fn: number; binary_accuracy: number | null; defect_recall: number | null; false_positive_rate: number | null; defect_precision: number | null; class_accuracy: number | null; mean_ms: number | null; median_ms: number | null; p95_ms: number | null }
type Summary = Metrics & { model_id: string; original: Metrics; transformed: Metrics }
type Run = { run_id: string; created_at: string; status: string; mode: string; models: SavedModel[]; samples: Sample[]; predictions: Prediction[]; summaries: Summary[]; error?: string }
type RunHeader = Pick<Run, 'run_id' | 'created_at' | 'status' | 'mode'>
type Listing = { models: SavedModel[]; active_job: ModelJob | null; runs: RunHeader[] }
const pct = (value: number | null | undefined) => value == null ? '—' : `${(100*value).toFixed(1)}%`
const ms = (value: number | null | undefined) => value == null ? '—' : `${value.toFixed(1)} ms`
const transformation = (s: Sample) => s.transformed ? `Rotated ${s.rotation_degrees}°` : 'Original · no transform'
const pipelineLabel = (m: SavedModel) => m.family === 'cnn' ? 'Segmentation → ResNet transforms' : m.family === 'classifiers' ? 'Full pipeline → features' : ({ normalized: 'Normalized full pipeline → features', full: 'Full pipeline → features', raw: 'Original image → features' }[String(m.config.preprocessing ?? 'full')] ?? String(m.config.preprocessing))
const modelType = (m: SavedModel) => m.family === 'anomaly' ? 'Anomaly detection' : m.family === 'cnn' ? 'CNN classifier' : 'Classifier'
const configLabel = (m: SavedModel) => Object.entries(m.config).filter(([k]) => ['variant', 'preprocessing', 'patch_size', 'variance_target', 'kernel', 'nu', 'gamma', 'C', 'n_neighbors', 'weights'].includes(k)).map(([k, v]) => `${k}: ${v}`).join(' · ')

export default function Comparison() {
  const [listing, setListing] = useState<Listing | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [count, setCount] = useState(12)
  const [mode, setMode] = useState<'single' | 'set'>('set')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [jobId, setJobId] = useState('')
  const [job, setJob] = useState<ModelJob | null>(null)
  const [runIds, setRunIds] = useState({ single: '', set: '' })
  const [runs, setRuns] = useState<Record<string, Run>>({})
  const runId = runIds[mode]
  const run = runs[runId]
  const history = listing?.runs.filter(r => r.mode === mode) ?? []
  const [metricGroup, setMetricGroup] = useState<'all' | 'original' | 'transformed'>('all')
  const initialized = useRef(false)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    comparisonRequest<Listing>('models/', undefined, controller.signal).then(data => {
      setListing(data)
      if (!initialized.current) {
        initialized.current = true
        setSelected(data.models.filter(m => m.available).map(m => m.id))
        setRunIds({ single: data.runs.find(r => r.mode === 'single')?.run_id ?? '', set: data.runs.find(r => r.mode === 'set')?.run_id ?? '' })
      }
      else setSelected(ids => ids.filter(id => data.models.some(m => m.id === id && m.available)))
      if (data.active_job) setJobId(data.active_job.job_id)
    }).catch(e => { if (!controller.signal.aborted) setError(String(e.message)) })
    return () => controller.abort()
  }, [refresh])
  useEffect(() => {
    if (!runId) return
    const controller = new AbortController()
    let timer: number | undefined
    const poll = async () => {
      try {
        const value = await comparisonRequest<Run>(`results/${runId}`, undefined, controller.signal)
        if (controller.signal.aborted) return
        setRuns(previous => ({ ...previous, [value.run_id]: value }))
        if (value.status === 'running') timer = window.setTimeout(poll, 1500)
      } catch (e) {
        if (!controller.signal.aborted) {
          if (jobId) timer = window.setTimeout(poll, 1500)
          else setError(e instanceof Error ? e.message : 'Could not load comparison')
        }
      }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [runId, jobId])
  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer: number | undefined
    const poll = async () => {
      try {
        const value = await getModelJob(jobId, controller.signal)
        if (controller.signal.aborted) return
        setJob(value)
        if (value.status === 'complete' || value.status === 'failed') {
          setJobId(''); setJob(null); setRefresh(r => r+1)
          if (value.error) setError(value.error)
        } else timer = window.setTimeout(poll, 1500)
      } catch (e) {
        if (!controller.signal.aborted) { setJobId(''); setJob(null); setError(e instanceof Error ? e.message : 'Job unavailable') }
      }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [jobId])
  const loadModels = async () => {
    setBusy(true); setError('')
    try {
      const value = await comparisonRequest<{ job_id: string }>('load/', { model_ids: selected })
      setJobId(value.job_id)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load models') }
    finally { setBusy(false) }
  }
  const start = async (mode: 'single' | 'set') => {
    setBusy(true); setError('')
    try {
      const value = await comparisonRequest<{ job_id: string; run_id: string }>('run/', { model_ids: selected, count: mode === 'single' ? 1 : count, mode })
      setJobId(value.job_id); setRunIds(previous => ({ ...previous, [mode]: value.run_id }))
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start comparison') }
    finally { setBusy(false) }
  }
  const locked = busy || !!jobId
  const ready = selected.length > 0 && selected.every(id => listing?.models.some(m => m.id === id && m.loaded))
  const sample = run?.samples[0]
  const visibleModels = run?.models.filter(m => selected.includes(m.id)) ?? []
  const summaries = visibleModels.map(m => {
    const row = run?.summaries.find(s => s.model_id === m.id)
    return { model: m, metrics: mode === 'single' || metricGroup === 'all' ? row : row?.[metricGroup] }
  })
  return <section className="comparison-page">
    <div className="page-intro model-writeup-intro"><h2>Comparison and Business Use</h2><ComparisonWriteup /></div>
    <section className="panel comparison-controls">
      <div className="panel-heading"><div><span className="comparison-eyebrow">01 / Models</span><h3>Choose your models</h3></div><div className="comparison-actions"><button disabled={locked || !selected.length} onClick={() => void loadModels()}>Load models</button><button onClick={() => { setError(''); setRefresh(r => r+1) }}>Refresh models</button></div></div>
      {!listing ? <p>Loading saved models…</p> : !listing.models.length ? <p>Use Save model in Anomaly Detection, Classifiers, or CNN to include a model here.</p> : <>
        <div className="model-controls"><button onClick={() => setSelected(listing.models.filter(m => m.available).map(m => m.id))}>Select all available</button><button onClick={() => setSelected([])}>Deselect all</button><span>{selected.length} selected</span></div>
        <div className="model-table-scroll"><table className="comparison-saved-table">
          <thead><tr><th>Select</th><th>Model name</th><th>Status</th><th>Type</th><th>Pipeline</th><th>Test images</th><th>Good/bad accuracy</th><th>Exact class accuracy</th><th>Defect recall</th><th>Missed defects</th><th>Good plates rejected</th><th>Macro F1</th></tr></thead>
          <tbody>{listing.models.map(m => <tr key={m.id}>
            <td><input id={`comparison-select-${m.id}`} type="checkbox" aria-label={`Select ${m.name}`} disabled={!m.available && !selected.includes(m.id)} checked={selected.includes(m.id)} onChange={e => {
              const checked = e.currentTarget.checked
              setSelected(ids => checked ? [...new Set([...ids, m.id])] : ids.filter(id => id !== m.id))
            }} /></td>
            <th scope="row"><label htmlFor={`comparison-select-${m.id}`}><strong>{m.name}</strong></label><small>{configLabel(m)}</small>{m.reason && <small role="note">{m.reason}</small>}</th>
            <td><span className={`comparison-badge ${m.loaded ? 'is-ready' : ''}`}>{m.loaded ? 'Loaded' : 'Not loaded'}</span></td><td>{modelType(m)}</td><td>{pipelineLabel(m)}</td><td>{m.metrics?.images ?? '—'}</td>
            <td>{pct(m.metrics?.binary_accuracy)}</td><td>{pct(m.metrics?.class_accuracy)}</td><td>{pct(m.metrics?.defect_recall)}</td><td>{m.metrics?.missed_defects ?? '—'}</td><td>{pct(m.metrics?.false_positive_rate)}</td><td>{pct(m.metrics?.macro_f1)}</td>
          </tr>)}</tbody>
        </table></div>
        <p className="feature-note">Metrics are from each model’s saved test evaluation, which may use different test splits. The shared random comparison results appear below. Unavailable metrics are shown as —.</p>
      </>}
      <p className="feature-note">Load your selected models once, then reuse them for image runs. Models stay in server memory until the server restarts; loading a new selection releases unchecked models. You can change the checkboxes at any time; unchecked models are hidden from the results below and excluded from the next run. A run already in progress keeps its original selection. Newly selected or changed models need to be loaded before running.</p>
    </section>
    {error && <p className="panel empty" role="alert">{error}</p>}
    {jobId && <p className="panel model-progress" role="status">{job?.phase ?? 'Preparing comparison…'} {job?.total ? `${job.done} / ${job.total}` : ''}</p>}
    <div className="comparison-activity-heading"><span className="comparison-eyebrow">02 / Evaluate</span><h3>Choose an activity</h3></div>
      <div className="view-switch comparison-tabs" role="tablist" aria-label="Comparison input mode">
        {(['single', 'set'] as const).map(value => <button key={value} id={`comparison-${value}-tab`} role="tab" aria-selected={mode === value} aria-pressed={mode === value} aria-controls="comparison-mode-panel" tabIndex={mode === value ? 0 : -1} onClick={() => setMode(value)} onKeyDown={event => {
          if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
            event.preventDefault()
            const next = event.key === 'Home' ? 'single' : event.key === 'End' ? 'set' : mode === 'single' ? 'set' : 'single'
            setMode(next)
            document.getElementById(`comparison-${next}-tab`)?.focus()
          }
        }}><strong>{value === 'single' ? 'Random Image' : 'Image Set'}</strong><span>{value === 'single' ? 'Inspect one plate and its predictions' : 'Compare models across a shared batch'}</span></button>)}
      </div>
      <div className="comparison-activity" id="comparison-mode-panel" role="tabpanel" aria-labelledby={`comparison-${mode}-tab`}>
        <section className="panel comparison-run-controls">
        <div className="panel-heading"><div><h3>{mode === 'single' ? 'Test a random plate' : 'Build a random image set'}</h3><p>{ready ? `${selected.length} selected models ready` : 'Select and load models above to begin.'}</p></div>
        <div className="model-controls">
          {mode === 'set' && <label>Image set size <input type="number" min="1" max="50" value={count} disabled={locked} onChange={e => setCount(Number(e.target.value))} /></label>}
          <button disabled={locked || !ready || (mode === 'set' && (!Number.isInteger(count) || count < 1 || count > 50))} onClick={() => void start(mode)}>{mode === 'set' ? 'Run on random image set' : 'Run on a single random image'}</button>
        </div></div>
        <p className="feature-note">{mode === 'set' ? 'Image sets sample only from the test folder without replacement, then independently rotate each image with 50% probability.' : 'Single-image runs first choose good or defective with a 50/50 draw, choose a test-folder image from that group, then independently choose whether to rotate it.'} All selected models receive the same generated inputs.</p>
      <details className="comparison-method"><summary>Sampling and rotation details</summary><p className="feature-note">Rotations expand the canvas and use a border-color fill to retain the image content. This approximates an orientation change by rotating the whole image. New runs use only the original test folder. Some classifiers and CNNs were trained using a shuffled split, so a test-folder image may still have been used in training; the per-model source label identifies known overlap.</p></details>
        </section>
    <section className="panel comparison-summary">
      <div className="panel-heading"><div><h3>{mode === 'single' ? 'Random image summary' : 'Image set summary'}</h3><p>Results for checked models in this activity’s selected run.</p></div>{!!history.length && <label className="comparison-history">Previous runs <select disabled={locked} value={runId} onChange={e => setRunIds(previous => ({ ...previous, [mode]: e.target.value }))}>{runId && !history.some(r => r.run_id === runId) && <option value={runId}>Current comparison</option>}{history.map(r => <option key={r.run_id} value={r.run_id}>{new Date(r.created_at).toLocaleString()} · {r.status}</option>)}</select></label>}</div>
      {run?.samples.some(s => s.original_split !== 'test') && <p className="feature-note">This saved run predates test-only sampling and includes training-folder images. Start a new run to evaluate test-folder images only.</p>}
      {mode === 'set' && <div className="view-switch">{(['all', 'original', 'transformed'] as const).map(value => <button key={value} aria-pressed={metricGroup === value} onClick={() => setMetricGroup(value)}>{value === 'all' ? 'All images' : value === 'original' ? 'Original images' : 'Rotated images'}</button>)}</div>}
      {run && !visibleModels.length && <p className="feature-note">No checked models have results in this run. Select models above or start a new run with your selection.</p>}
      {!run ? <p>Run a comparison to see metrics and timings.</p> : <><div className="comparison-run-stats"><span><strong>{run.samples.length}</strong> images</span><span><strong>{run.samples.filter(s => s.actual === 'good').length}</strong> good</span><span><strong>{run.samples.filter(s => s.actual !== 'good').length}</strong> defective</span><span><strong>{run.samples.filter(s => s.transformed).length}</strong> rotated</span><span className="comparison-badge">{run.status}</span></div><div className="model-table-scroll"><table><thead><tr><th>Model</th><th>Evaluated / errors</th><th>Good/bad accuracy</th><th>Exact class accuracy</th><th>Defects caught</th><th>Missed defects</th><th>Good plates rejected</th><th>Defect precision</th><th>Mean time</th><th>Median time</th><th>P95 time</th></tr></thead><tbody>{summaries.map(({ model: m, metrics: x }) => <tr key={m.id}><th>{m.name}</th><td>{x ? `${x.evaluated} / ${x.errors}` : 'Waiting'}</td><td>{pct(x?.binary_accuracy)}</td><td>{pct(x?.class_accuracy)}</td><td>{x ? `${x.tp}/${x.tp+x.fn} (${pct(x.defect_recall)})` : '—'}</td><td>{x?.fn ?? '—'}</td><td>{x ? `${x.fp}/${x.fp+x.tn} (${pct(x.false_positive_rate)})` : '—'}</td><td>{pct(x?.defect_precision)}</td><td>{ms(x?.mean_ms)}</td><td>{ms(x?.median_ms)}</td><td>{ms(x?.p95_ms)}</td></tr>)}</tbody></table></div></>}
      <details className="comparison-method"><summary>How prediction time is measured</summary><p className="feature-note">Timing uses CPU inference with one compute thread, after an untimed warm-up when a loaded model is first used. Later runs reuse the loaded, warmed model. It includes preprocessing, features when applicable, and the model decision; it excludes loading weights, reading files, generating rotations, and explanation plots. Errors are counted separately and excluded from accuracy and timing aggregates. Exact class accuracy does not apply to binary models.</p></details>
      {run?.models.some(m => m.warmup_succeeded === false) && <p className="feature-note">At least one model could not warm up on the first image; its timings may include initialization overhead.</p>}
    </section>
    <section className="panel comparison-images">
      <div className="panel-heading"><div><h3>{mode === 'single' ? 'Individual prediction' : 'Image set predictions'}</h3><p>{mode === 'single' ? 'Inspect the plate alongside each model’s decision.' : 'Each row is one generated image; each column is one model.'}</p></div></div>
      {run && sample ? mode === 'single' ? <div className="comparison-inspection">
        <figure><div className="comparison-image-stage"><img src={comparisonImageUrl(run.run_id, sample.id)} alt={`${sample.actual}, ${transformation(sample)}`} /></div><figcaption><strong>{sample.actual}</strong><span className="comparison-badge">{transformation(sample)}</span><small>{sample.path} · original {sample.original_split} folder</small></figcaption></figure>
        <div className="model-table-scroll"><table><thead><tr><th>Model</th><th>Prediction</th><th>Correct?</th><th>Time</th><th>Source used for this model</th></tr></thead><tbody>{visibleModels.map(m => { const p = run.predictions.find(r => r.model_id === m.id && r.sample_id === sample.id); return <tr key={m.id}><th>{m.name}</th><td>{p?.error ?? p?.prediction ?? 'Waiting'}</td><td className={p?.error ? '' : p?.correct === true ? 'classification-right' : p?.correct === false ? 'classification-wrong' : ''}>{p?.error ? 'Error' : p ? `${p.correct ? 'Right' : 'Wrong'}${p.supports_classes ? ' class' : ' good/bad'}` : '—'}</td><td>{ms(p?.elapsed_ms)}</td><td>{p?.exposure ?? '—'}</td></tr> })}</tbody></table></div>
      </div> : <div className="model-table-scroll comparison-batch-scroll"><table className="comparison-batch-table">
        <thead><tr><th>Image / actual class</th><th>Transformation</th>{visibleModels.map(m => <th key={m.id}>{m.name}</th>)}</tr></thead>
        <tbody>{run.samples.map(s => <tr key={s.id}>
          <th scope="row"><div className="comparison-image-cell"><a href={comparisonImageUrl(run.run_id, s.id)} target="_blank" rel="noreferrer"><img loading="lazy" src={comparisonImageUrl(run.run_id, s.id)} alt={`${s.actual}, ${transformation(s)}`} /></a><span><strong>{s.actual}</strong><small>{s.path}</small><small>Original {s.original_split} folder</small></span></div></th>
          <td><span className="comparison-badge">{transformation(s)}</span></td>
          {visibleModels.map(m => { const p = run.predictions.find(r => r.model_id === m.id && r.sample_id === s.id); return <td key={m.id}><strong>{p?.error ?? p?.prediction ?? 'Waiting'}</strong><small className={p?.error ? '' : p?.correct === true ? 'classification-right' : p?.correct === false ? 'classification-wrong' : ''}>{p?.error ? 'Error' : p ? `${p.correct ? 'Right' : 'Wrong'}${p.supports_classes ? ' class' : ' good/bad'}` : '—'}</small><small>{ms(p?.elapsed_ms)} · {p?.exposure ?? '—'}</small></td> })}
        </tr>)}</tbody>
      </table></div> : <div className="comparison-empty"><strong>{mode === 'single' ? 'No random image yet' : 'No image set yet'}</strong><p>{mode === 'single' ? 'Run a single image to inspect its class, rotation, and predictions.' : 'Generate an image set to compare all selected models in one table.'}</p></div>}
    </section>
    </div>
  </section>
}
