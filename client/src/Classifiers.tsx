import { useEffect, useState } from 'react'
import ClassifierLearningCurves from './ClassifierLearningCurves'
import RandomClassifierReview from './RandomClassifierReview'
import { classifierRequest, getImageUrl, getModelJob } from './api'
import type { ModelJob } from './api'

type Sample = { path: string; label: string; original_split: string }
type Evaluation = { accuracy: number; balanced_accuracy: number; classes: string[]; confusion_matrix: number[][]; predictions: string[]; report: Record<string, Metric> }
type Metric = { precision: number; recall: number; 'f1-score': number; support: number }
export type Model = { id: string; config: { kind: string; variance_target: number; [key: string]: string | number }; cv: Record<string, { mean: number; std: number; folds?: number[] }>; evaluation: Evaluation; training_evaluation?: Evaluation }
export type Summary = { run_id: string; seed: number; folds: number; train: Sample[]; test: Sample[]; classes: string[]; models: Model[]; failures: { error: string }[] }
type Listing = { summary: Summary | null; active_job: ModelJob | null; configurations: number }
type Plot = { training: (Sample & { point: number[] })[]; selected: Sample & { point: number[]; prediction: string }; neighbors: (Sample & { index: number; distance: number })[]; support_indices: number[]; regions: string[][]; bounds: number[][]; variance: number[]; dimensions: number }
const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const gap = (train: number | undefined, test: number) => train === undefined ? '—' : `${train - test > 0 ? '+' : ''}${((train - test) * 100).toFixed(1)} pp`
const name = (m: Model) => Object.entries(m.config).map(([k, v]) => k === 'kind' ? String(v).toUpperCase() : `${k}=${v}`).join(' · ')
const colors = ['#278568', '#be433e', '#426bcd', '#aa56ad', '#c28717', '#208b91', '#686868']

function Space({ model, index, classes }: { model: Model; index: number; classes: string[] }) {
  const [plot, setPlot] = useState<Plot | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    classifierRequest<Plot>(`inspect/?model_id=${model.id}&image_index=${index}`, undefined, controller.signal).then(setPlot).catch((e: unknown) => { if (!controller.signal.aborted) setError(String(e)) })
    return () => controller.abort()
  }, [model.id, index])
  if (error) return <p role="alert">{error}</p>
  if (!plot) return <p role="status">Loading decision space…</p>
  const color = (label: string) => colors[classes.indexOf(label) % colors.length]
  const [low, high] = plot.bounds
  const x = (v: number) => 45 + (v - low[0]) / (high[0] - low[0]) * 500
  const y = (v: number) => 345 - (v - low[1]) / (high[1] - low[1]) * 300
  const size = plot.regions.length
  const neighbors = new Set(plot.neighbors.map((p) => p.index))
  const supports = new Set(plot.support_indices)
  return <div className="classifier-space"><h3>{model.config.kind.toUpperCase()} training space</h3>
    <svg viewBox="0 0 590 390" role="img" aria-label="Fixed PCA projection of training points and selected test image">
      {plot.regions.flatMap((row, j) => row.map((label, i) => <rect key={`${i}:${j}`} x={45 + i / size * 500} y={45 + (size - 1 - j) / size * 300} width={500 / size} height={300 / size} fill={color(label)} opacity=".2" shapeRendering="crispEdges" />))}
      {plot.neighbors.map((p) => <line key={p.index} x1={x(plot.selected.point[0])} y1={y(plot.selected.point[1])} x2={x(plot.training[p.index].point[0])} y2={y(plot.training[p.index].point[1])} stroke="#1c2933" strokeDasharray="3 3" />)}
      {plot.training.map((p, i) => <circle key={p.path} cx={x(p.point[0])} cy={y(p.point[1])} r={neighbors.has(i) ? 6 : 3.5} fill={color(p.label)} stroke={neighbors.has(i) || supports.has(i) ? '#18242b' : 'white'} strokeWidth={neighbors.has(i) ? 2 : .8}><title>{p.path} · {p.label}{supports.has(i) ? ' · support vector' : ''}</title></circle>)}
      <path d={`M${x(plot.selected.point[0])},${y(plot.selected.point[1])-9} l9,9 l-9,9 l-9,-9 Z`} fill={color(plot.selected.prediction)} stroke="#111" strokeWidth="2"><title>{plot.selected.path} · predicted {plot.selected.prediction}</title></path>
      <text x="290" y="380" textAnchor="middle">PC1 · {pct(plot.variance[0])} variance</text><text x="45" y="24">PC2 · {pct(plot.variance[1] ?? 0)} variance</text>
      {[0, .5, 1].map((f) => <g key={f}><text x={45+f*500} y="361" textAnchor="middle">{(low[0]+f*(high[0]-low[0])).toFixed(1)}</text><text x="40" y={349-f*300} textAnchor="end">{(low[1]+f*(high[1]-low[1])).toFixed(1)}</text></g>)}
    </svg>
    <div className="classifier-legend">{classes.map((label) => <span key={label}><i style={{ background: color(label) }} />{label}</span>)}</div>
    <p className="feature-note">Circles: training images. Diamond: selected test image, colored by prediction. Axes and training points stay fixed for this model. Display bounds include all saved holdout images, but only the selected image is shown. Background colors show a fixed decision slice with higher PCs held at their training means. A projected point can differ from its background color because predictions use all PCs. The model uses {plot.dimensions} PCs.</p>
    {model.config.kind === 'svm' && <p className="feature-note">Outlined circles are support vectors in the full fitted SVM.</p>}
    {!!plot.neighbors.length && <div className="model-table-scroll"><table><caption>Actual neighbors in all retained PCs; lines show their 2D projections</caption><thead><tr><th>Image</th><th>Class</th><th>Distance</th></tr></thead><tbody>{plot.neighbors.map((p) => <tr key={p.path}><td>{p.path.split('/').pop()}</td><td>{p.label}</td><td>{p.distance.toFixed(3)}</td></tr>)}</tbody></table></div>}
  </div>
}

function PerformanceOverview({ models, classes, selected, onSelect }: { models: Model[]; classes: string[]; selected?: string; onSelect: (id: string) => void }) {
  const ranked = [...models].sort((a, b) => b.cv.macro_f1.mean - a.cv.macro_f1.mean || name(a).localeCompare(name(b)))
  return <div className="classifier-performance">
    <section className="panel"><div className="panel-heading"><h3>Class recall heatmap</h3><span>Holdout · 0–100%</span></div>
      <p className="feature-note">Rows follow CV macro-F1 rank. Darker green means more images of that actual class were correctly classified. Cells show correct/total; hover for recall.</p>
      <div className="model-table-scroll classifier-performance-scroll"><table><thead><tr><th>Model</th>{classes.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{ranked.map((m, i) => <tr key={m.id} aria-selected={selected === m.id}><th><button title={name(m)} onClick={() => onSelect(m.id)}>{i + 1}. {name(m)}</button></th>{classes.map(c => {
        const metric = m.evaluation.report[c]
        const recall = metric?.recall ?? 0
        const total = metric?.support ?? 0
        return <td key={c} style={{ background: total ? `hsl(151 38% ${96 - recall * 67}%)` : '#eee', color: total && recall > .6 ? '#fff' : '#182c22' }} title={`${c}: ${total ? pct(recall) : 'No holdout samples'}`} aria-label={`${c}: ${total ? pct(recall) + ' recall' : 'no samples'}`}>{total ? `${Math.round(recall * total)}/${total} (${pct(recall)})` : '—'}</td>
      })}</tr>)}</tbody></table></div>
    </section>
    <section className="panel"><div className="panel-heading"><h3>Cross-validation ranking</h3><span>Macro F1 · training folds</span></div>
      <p className="feature-note">Ranked by mean macro F1. Dots are individual folds; the green diamond is the mean. The line shows ±1 standard deviation, not a confidence interval. Click a model to inspect it below.</p>
      <div className="model-table-scroll classifier-performance-scroll"><table><thead><tr><th>Model</th><th>Fold scores · 0–100%</th><th>Mean ± SD</th></tr></thead><tbody>{ranked.map((m, i) => {
        const score = m.cv.macro_f1
        const x = (v: number) => 10 + Math.max(0, Math.min(1, v)) * 220
        return <tr key={m.id} aria-selected={selected === m.id}><th><button title={name(m)} onClick={() => onSelect(m.id)}>{i + 1}. {name(m)}</button></th><td><svg className="classifier-cv-score" viewBox="0 0 240 48" role="img" aria-label={`${name(m)}: mean ${pct(score.mean)}, standard deviation ${pct(score.std)}`}>
          {[0, .25, .5, .75, 1].map(v => <g key={v}><line x1={x(v)} x2={x(v)} y1="5" y2="30" stroke="#dce5df" /><text x={x(v)} y="44" textAnchor="middle">{v * 100}</text></g>)}
          <line x1={x(score.mean-score.std)} x2={x(score.mean+score.std)} y1="17" y2="17" stroke="#235d44" strokeWidth="3" />
          {(score.folds ?? []).map((v, j) => <circle key={j} cx={x(v)} cy={9 + (j % 3) * 8} r="3" fill="#486bd4"><title>Fold {j + 1}: {pct(v)}</title></circle>)}
          <path d={`M${x(score.mean)},11 l6,6 l-6,6 l-6,-6 Z`} fill="#235d44"><title>Mean: {pct(score.mean)}</title></path>
        </svg></td><td>{pct(score.mean)} ± {pct(score.std)}</td></tr>
      })}</tbody></table></div>
    </section>
  </div>
}

function Generalization({ model }: { model: Model }) {
  const training = model.training_evaluation
  const metrics = [
    { label: 'Accuracy', train: training?.accuracy, test: model.evaluation.accuracy, cv: model.cv.accuracy },
    { label: 'Balanced accuracy', train: training?.balanced_accuracy, test: model.evaluation.balanced_accuracy, cv: model.cv.balanced_accuracy },
    { label: 'Macro F1', train: training?.report['macro avg']['f1-score'], test: model.evaluation.report['macro avg']['f1-score'], cv: model.cv.macro_f1 },
    { label: 'Weighted F1', train: training?.report['weighted avg']['f1-score'], test: model.evaluation.report['weighted avg']['f1-score'] },
  ]
  return <section className="classifier-generalization"><h3>Training vs validation vs holdout</h3>
    <p>Positive training-minus-holdout gaps mean performance dropped on unseen images. Large gaps can indicate overfitting, but also reflect split variability. Training scores are measured on the fitted model’s own examples; distance-weighted KNN can score perfectly by matching an image to itself. Use CV and holdout scores to judge generalization.</p>
    <div className="model-table-scroll"><table><thead><tr><th>Metric</th><th>Training</th><th>CV validation mean ± SD</th><th>Holdout</th><th>Training − holdout</th></tr></thead><tbody>{metrics.map(m => <tr key={m.label}><th>{m.label}</th><td>{m.train === undefined ? '—' : pct(m.train)}</td><td>{m.cv ? `${pct(m.cv.mean)} ± ${pct(m.cv.std)}` : '—'}</td><td>{pct(m.test)}</td><td>{gap(m.train,m.test)}</td></tr>)}</tbody></table></div>
  </section>
}

export default function Classifiers() {
  const [trainingMetrics, setTrainingMetrics] = useState<{ run: string; values: Record<string, Evaluation> }>({ run: '', values: {} })
  const [metricsError, setMetricsError] = useState('')
  const [metricsAttempt, setMetricsAttempt] = useState(0)
  const [data, setData] = useState<Listing | null>(null)
  const [job, setJob] = useState<ModelJob | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [family, setFamily] = useState('all')
  const [choice, setChoice] = useState('')
  const [index, setIndex] = useState(0)
  const [sort, setSort] = useState({ key: 'cv', descending: true })
  useEffect(() => {
    const controller = new AbortController()
    let timer: number
    const poll = async () => {
      try {
        const listing = await classifierRequest<Listing>('', undefined, controller.signal)
        if (controller.signal.aborted) return
        setData(listing)
        if (listing.active_job) setJob(listing.active_job)
      } catch (e) { if (!controller.signal.aborted) setError(String(e)) }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 5000)
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [])
  const jobId = job?.job_id
  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer: number
    const poll = async () => {
      try {
        const value = await getModelJob(jobId, controller.signal)
        if (controller.signal.aborted) return
        if (value.status === 'complete' || value.status === 'failed') {
          setJob(null)
          if (value.error) setError(value.error)
          const listing = await classifierRequest<Listing>('', undefined, controller.signal)
          if (!controller.signal.aborted) setData(listing)
        } else { setJob(value); timer = window.setTimeout(poll, 1000) }
      } catch (e) { if (!controller.signal.aborted) { setError(String(e)); setJob(null) } }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [jobId])
  const run = data?.summary?.run_id
  const modelCount = data?.summary?.models.length ?? 0
  useEffect(() => {
    if (!run || !modelCount) return
    const controller = new AbortController()
    classifierRequest<Record<string, Evaluation>>(`training_metrics/?run_id=${run}`, undefined, controller.signal).then(values => {
      if (!controller.signal.aborted) { setTrainingMetrics({ run, values }); setMetricsError('') }
    }).catch((e: unknown) => { if (!controller.signal.aborted) setMetricsError(String(e)) })
    return () => controller.abort()
  }, [run, modelCount, metricsAttempt])
  const summary = data?.summary
  const testIndex = summary?.test[index] ? index : 0
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || !['ArrowLeft','ArrowRight'].includes(e.key) || (e.target instanceof Element && e.target.closest('input,select,textarea,[contenteditable],.random-classifier-review'))) return
      e.preventDefault(); setIndex((current) => Math.max(0, Math.min((summary?.test.length ?? 1)-1, current + (e.key === 'ArrowLeft' ? -1 : 1))))
    }
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler)
  }, [summary?.test.length])
  const value = (m: Model, key: string): number | string => key === 'name' ? name(m) : key === 'cv' ? m.cv.macro_f1.mean : key === 'train_accuracy' ? m.training_evaluation?.accuracy ?? -Infinity : key === 'accuracy_gap' ? m.training_evaluation ? m.training_evaluation.accuracy - m.evaluation.accuracy : -Infinity : key === 'f1_gap' ? m.training_evaluation ? m.training_evaluation.report['macro avg']['f1-score'] - m.evaluation.report['macro avg']['f1-score'] : -Infinity : key === 'accuracy' ? m.evaluation.accuracy : key === 'balanced' ? m.evaluation.balanced_accuracy : key === 'f1' ? m.evaluation.report['macro avg']['f1-score'] : m.evaluation.report[key]?.recall ?? 0
  const rows = (summary?.models ?? []).map(m => ({ ...m, training_evaluation: m.training_evaluation ?? (trainingMetrics.run === run ? trainingMetrics.values[m.id] : undefined) })).filter((m) => family === 'all' || m.config.kind === family).sort((a,b) => { const av=value(a,sort.key), bv=value(b,sort.key); return (typeof av === 'string' ? av.localeCompare(String(bv)) : Number(av)-Number(bv)) * (sort.descending ? -1 : 1) })
  const model = rows.find((m) => m.id === choice) ?? rows[0]
  const sample = summary?.test[testIndex]
  const train = async () => {
    setBusy(true); setError('')
    try { const result = await classifierRequest<{ job_id: string }>('train/', { seed: 42 }); setJob({ ...result, kind: 'classifiers', status: 'queued', phase: 'Queued', done: 0, total: 0 }) }
    catch(e) { setError(String(e)) } finally { setBusy(false) }
  }
  const header = (key: string, label: string) => <th key={key}><button onClick={() => setSort((s) => ({ key, descending: s.key === key ? !s.descending : true }))}>{label}{sort.key === key ? sort.descending ? ' ↓' : ' ↑' : ''}</button></th>
  return <section><div className="page-intro"><h2>Classifiers</h2><p>Predict good plates and each defect class using LAB, Sobel, HOG, and Frangi. A shared mixed 80/20 split keeps the holdout separate from cross-validation. Seed 42 · 64-pixel patches · 918 pooled image features.</p></div>
    <div className="panel model-training"><button disabled={busy || !!job} onClick={() => void train()}>Train all classifiers ({data?.configurations ?? 42})</button><p>PCA: 90/95/99% variance × Euclidean/Manhattan centroids. SVM: same variances × linear/RBF × C 0.1/1/10. KNN: same variances × 3/5/9 neighbors × uniform/distance weights. Previous runs stay saved on disk.</p></div>
    {error && <p className="panel empty" role="alert">{error}</p>}
    {job && <p className="panel model-progress" role="status">{job.phase} · {job.done}/{job.total}{job.combinations ? ` · ${job.combination}/${job.combinations} combinations processed` : ''}</p>}
    {!summary ? <p className="panel empty">Train classifiers to create a shared split and comparisons.</p> : <>
      <section className="panel"><div className="panel-heading"><h3>Data split</h3><span>{summary.folds}-fold grouped stratified CV · training only</span></div><p className="feature-note">Original train and test images are pooled. Exact duplicates stay together. Class rounding/group sizes can make the realized fraction differ slightly from 80/20.</p><div className="model-table-scroll"><table><thead><tr><th>Class</th><th>Training</th><th>Holdout</th><th>Total</th><th>Training %</th></tr></thead><tbody>{[...summary.classes, 'All classes'].map((label) => { const a=summary.train.filter(s => label==='All classes'||s.label===label).length,b=summary.test.filter(s => label==='All classes'||s.label===label).length;return <tr key={label}><th>{label}</th><td>{a}</td><td>{b}</td><td>{a+b}</td><td>{pct(a/(a+b))}</td></tr> })}</tbody></table></div></section>
      <nav className="model-tabs">{['all','pca','svm','knn'].map((f) => <button key={f} aria-pressed={family===f} onClick={() => setFamily(f)}>{f.toUpperCase()}</button>)}</nav>
      <PerformanceOverview models={rows} classes={summary.classes} selected={model?.id} onSelect={setChoice} />
      <section className="panel"><div className="panel-heading"><h3>Classifier comparison</h3><span>{rows.length} models</span></div><p className="feature-note">Default ranking uses cross-validation macro F1, not holdout scores. CV values are mean ± standard deviation. Gaps are training minus holdout in percentage points (pp). Click a model to inspect its metrics and predictions.</p><div className="model-table-scroll"><table><thead><tr>{header('name','Model')}{header('cv','CV macro F1')}{header('train_accuracy','Training accuracy')}{header('accuracy','Holdout accuracy')}{header('accuracy_gap','Accuracy gap')}{header('f1_gap','Macro-F1 gap')}{header('balanced','Balanced accuracy')}{header('f1','Macro F1')}{summary.classes.map(c => header(c,c))}</tr></thead><tbody>{rows.map(m => <tr key={m.id} aria-selected={model?.id===m.id}><th><button onClick={() => setChoice(m.id)}>{name(m)}</button></th><td>{pct(m.cv.macro_f1.mean)} ± {pct(m.cv.macro_f1.std)}</td><td>{m.training_evaluation ? pct(m.training_evaluation.accuracy) : '—'}</td><td>{pct(m.evaluation.accuracy)}</td><td>{gap(m.training_evaluation?.accuracy,m.evaluation.accuracy)}</td><td>{gap(m.training_evaluation?.report['macro avg']['f1-score'],m.evaluation.report['macro avg']['f1-score'])}</td><td>{pct(m.evaluation.balanced_accuracy)}</td><td>{pct(m.evaluation.report['macro avg']['f1-score'])}</td>{summary.classes.map(c=>{const r=m.evaluation.report[c];return <td key={c}>{Math.round(r.recall*r.support)}/{r.support} ({pct(r.recall)})</td>})}</tr>)}</tbody></table></div></section>
      {metricsError && <p className="feature-note" role="alert">Training metrics: {metricsError} <button onClick={() => setMetricsAttempt(a => a+1)}>Retry</button></p>}
      {!metricsError && rows.some(m => !m.training_evaluation) && <p className="feature-note" role="status">Calculating training metrics from saved models… No retraining is needed.</p>}
      {!!summary.failures.length && <details className="panel model-training"><summary>{summary.failures.length} failed configurations</summary>{summary.failures.map((f,i)=><p key={i}>{f.error}</p>)}</details>}
      {model && <section className="panel model-training"><h3>Evaluation · {name(model)}</h3><Generalization model={model} /><ClassifierLearningCurves key={`${summary.run_id}:${model.id}`} run={summary.run_id} model={model.id} /><p>Balanced accuracy averages class recall. Macro F1 weights classes equally; weighted F1 accounts for class frequency. Weighted F1: {pct(model.evaluation.report['weighted avg']['f1-score'])}.</p><div className="model-composition-grid"><div className="model-table-scroll"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{summary.classes.map(c=>{const r=model.evaluation.report[c];return <tr key={c}><th>{c}</th><td>{pct(r.precision)}</td><td>{pct(r.recall)}</td><td>{pct(r['f1-score'])}</td><td>{r.support}</td></tr>})}</tbody></table></div><div className="model-table-scroll"><table><caption>Confusion matrix · rows actual, columns predicted</caption><thead><tr><th>Actual / predicted</th>{model.evaluation.classes.map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>{model.evaluation.confusion_matrix.map((r,i)=><tr key={i}><th>{model.evaluation.classes[i]}</th>{r.map((n,j)=><td key={j} style={{ background: i===j ? '#e1f0e6' : n ? '#fce2df' : undefined }}>{n}</td>)}</tr>)}</tbody></table></div></div></section>}
      {sample && <section className="panel"><div className="panel-heading"><h3>Individual holdout predictions</h3><div className="model-controls"><button disabled={testIndex===0} onClick={()=>setIndex(testIndex-1)}>←</button><select aria-label="Holdout image" value={testIndex} onChange={e=>setIndex(Number(e.target.value))}>{summary.test.map((s,i)=><option key={s.path} value={i}>{s.path}</option>)}</select><button disabled={testIndex===summary.test.length-1} onClick={()=>setIndex(testIndex+1)}>→</button></div></div><div className="classifier-inspection"><figure><img src={getImageUrl('metal_plate', sample.original_split, sample.path.split('/').slice(2).join('/'))} alt={sample.path}/><figcaption>{sample.path.split('/').pop()} · actual: {sample.label}{model && <p className={model.evaluation.predictions[testIndex]===sample.label ? 'classification-right' : 'classification-wrong'}>{model.evaluation.predictions[testIndex]===sample.label ? 'Right' : 'Wrong'} · predicted: {model.evaluation.predictions[testIndex]}</p>}</figcaption></figure><div className="model-table-scroll"><table><thead><tr><th>Model</th><th>Prediction</th><th>Correct</th></tr></thead><tbody>{[...rows].sort((a,b)=>Number(b.evaluation.predictions[testIndex]===sample.label)-Number(a.evaluation.predictions[testIndex]===sample.label)).map(m=><tr key={m.id} aria-selected={model?.id===m.id}><th><button onClick={()=>setChoice(m.id)}>{name(m)}</button></th><td>{m.evaluation.predictions[testIndex]}</td><td className={m.evaluation.predictions[testIndex]===sample.label?'classification-right':'classification-wrong'}>{m.evaluation.predictions[testIndex]===sample.label?'Right':'Wrong'}</td></tr>)}</tbody></table></div>{model && <Space key={`${model.id}:${testIndex}`} model={model} index={testIndex} classes={summary.classes}/>}</div></section>}
      {summary.run_id && !!summary.models.length && <RandomClassifierReview key={`${summary.run_id}:${summary.models.length}`} summary={summary} training={!!job || !!data?.active_job} />}
    </>}
  </section>
}
