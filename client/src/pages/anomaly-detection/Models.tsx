import { AnomalyWriteup } from './AnomalyWriteup'
import { useEffect, useState } from 'react'
import SaveModelButton from '../../components/SaveModelButton'
import PatchOverlay from '../../components/PatchOverlay'
import SortHeader from '../../components/SortHeader'
import { useModelSort } from '../../hooks/useModelSort'
import { getModels, getPCAModel, getModelJob, startPCAJob, getImages, getImageUrl, getModelProjection } from '../../api'
import type { ModelJob, PCAModel, PCAReport, PCAProjection, AnomalyPreprocessing } from '../../api'

const percent = (v: number | null | undefined) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
const number = (v: number) => v !== 0 && Math.abs(v) < .001 ? v.toExponential(3) : v.toLocaleString(undefined, { maximumFractionDigits: 3 })
const pipelineNames = { full: 'Full · no normalization', normalized: 'Full · normalization', raw: 'Raw · features only', legacy: 'Legacy pipeline' }
const runName = (r: PCAReport) => `${r.config.model_type === 'one_class_svm' ? `SVM ${r.config.kernel} · ν ${r.config.nu} · γ ${r.config.gamma}` : `PCA ${r.config.distance_metric ?? 'squared_l2'}`} · ${r.config.patch_size}px · ${percent(r.config.variance_target)} · ${pipelineNames[r.config.preprocessing ?? 'legacy']} · ${r.model_id.slice(0, 6)}`

function useModel(id: string) {
  const [state, setState] = useState<{ id: string; data?: PCAModel; error?: string }>({ id: '' })
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    getPCAModel(id, controller.signal).then((data) => { if (!controller.signal.aborted) setState({ id, data }) }).catch((e: unknown) => {
      if (!controller.signal.aborted) setState({ id, error: e instanceof Error ? e.message : 'Could not load model' })
    })
    return () => controller.abort()
  }, [id])
  return state.id === id ? state : { id }
}

function Projection({ report, selected, onProjection }: { report?: PCAReport; selected: string; onProjection: (value: { key: string; data?: PCAProjection; error?: string }) => void }) {
  const id = report?.model_id ?? ''
  const model = useModel(id)
  const key = `${id}:${selected}`
  const [state, setState] = useState<{ key: string; data?: PCAProjection; error?: string }>({ key: '' })
  useEffect(() => {
    if (!id || !selected || (report?.compatible === false && !report?.results_saved)) return
    const controller = new AbortController()
    getModelProjection(id, `metal_plate/test/${selected}`, controller.signal).then((data) => {
      if (!controller.signal.aborted) { setState({ key, data }); onProjection({ key, data }) }
    }).catch((e: unknown) => { if (!controller.signal.aborted) { const error = e instanceof Error ? e.message : 'Could not project image'; setState({ key, error }); onProjection({ key, error }) } })
    return () => controller.abort()
  }, [id, selected, key, report?.compatible, report?.results_saved, onProjection])
  if (!report) return <p className="empty">Train models to inspect their PCA space.</p>
  return <div><p className="feature-note">{runName(report)}</p>
    {model.error && <p role="alert">{model.error}</p>}
    {(report.compatible === false && !report.results_saved) ? <p className="feature-note">Retrain this model for the current pipeline. Only its saved reference space is shown.</p> : state.key !== key ? <p className="feature-note" role="status">Projecting selected image…</p> : state.error ? <p role="alert" className="feature-note">{state.error}</p> : <p className="feature-note">{state.data?.prediction} · score {number(state.data?.plate_score ?? 0)}</p>}
    {model.data && <PCAScatter key={id} model={model.data} test={state.key === key ? state.data ?? null : null} />}
  </div>
}

function Variance({ model }: { model: PCAModel }) {
  const variance = model.explained_variance_ratio
  return <article className="model-chart pca-variance"><h3>Total variance</h3><svg viewBox="0 0 620 280" role="img" aria-label="PCA scree and cumulative explained variance">
    {variance.map((v, i) => { const cumulative = variance.slice(0, i + 1).reduce((sum, x) => sum + x, 0); return <g key={i}><rect x={45 + i / variance.length * 550} y={220 - v * 180} width={Math.max(1, 550 / variance.length - 1)} height={v * 180} fill={i < model.components ? '#278568' : '#bccdc3'}><title>PC{i + 1}: {percent(v)}</title></rect><circle cx={45 + (i + .5) / variance.length * 550} cy={220 - cumulative * 180} r="2" fill="#486bd4"><title>Cumulative PC{i + 1}: {percent(cumulative)}</title></circle></g> })}
    {[0, .5, 1].map((v) => <text key={v} x="37" y={224 - v * 180} textAnchor="end">{v * 100}%</text>)}<text x="45" y="247">PC1</text><text x="590" y="247" textAnchor="end">PC{variance.length}</text><text x="310" y="269" textAnchor="middle">Bars: individual variance · Blue: cumulative</text>
  </svg><p>{model.features} features → {model.components} components · {percent(model.retained_variance)} retained variance.</p></article>
}

function PCAAnalysis({ reports }: { reports: PCAReport[] }) {
  const [choice, setChoice] = useState('')
  const id = reports.some((r) => r.model_id === choice) ? choice : reports[0]?.model_id ?? ''
  const index = reports.findIndex((r) => r.model_id === id)
  const model = useModel(id)
  const move = (offset: number) => { if (reports[index + offset]) setChoice(reports[index + offset].model_id) }
  return <section className="panel pca-analysis" tabIndex={0} aria-label="PCA analysis model navigation" onKeyDown={(e) => {
    if (e.altKey || e.metaKey || e.ctrlKey || e.shiftKey || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return
    if (e.target instanceof Element && e.target.closest('input, textarea, select:not([data-analysis-model]), [contenteditable="true"]')) return
    e.preventDefault(); move(e.key === 'ArrowLeft' ? -1 : 1)
  }}><div className="model-training"><h3>PCA analysis</h3><div className="model-controls"><button disabled={index <= 0} onClick={() => move(-1)}>← Previous model</button><label>Model <select data-analysis-model value={id} disabled={!reports.length} onChange={(e) => setChoice(e.target.value)}>{reports.map((r) => <option key={r.model_id} value={r.model_id}>{runName(r)}</option>)}</select></label><button disabled={index < 0 || index >= reports.length - 1} onClick={() => move(1)}>Next model →</button><span>{index >= 0 ? index + 1 : 0} / {reports.length}</span></div><p>Choose a model, then use ← / → to compare models. Component weights below show where each model’s variance comes from.</p></div>
    {model.error && <p role="alert">{model.error}</p>}{id && !model.data && !model.error && <p className="empty">Loading PCA analysis…</p>}
    {model.data && <><Variance model={model.data} /><ComponentContributions key={id} model={model.data} /></>}
  </section>
}
function PCAScatter({ model, test }: { model: PCAModel; test: PCAProjection | null }) {
  const [axes, setAxes] = useState([0, Math.min(1, model.components - 1)])
  const [hover, setHover] = useState('')
  const step = Math.max(1, Math.ceil(model.training_points.length / 1600))
  const training = model.training_points.filter((_, index) => index % step === 0)
  const points = [...training.map((p) => ({ ...p, group: 'Training good', color: '#278568' })),
    ...model.calibration_points.map((p) => ({ ...p, group: 'Held-out good', color: '#d99722' })),
    ...(test?.patches.map((p) => ({ ...p, group: p.anomalous ? 'Test · anomalous patch' : 'Test · normal patch', color: p.anomalous ? '#c4373d' : '#366ac7' })) ?? [])]
  const xs = points.map((p) => p.scores[axes[0]])
  const ys = points.map((p) => p.scores[axes[1]])
  const lowX = Math.min(...xs), highX = Math.max(...xs), lowY = Math.min(...ys), highY = Math.max(...ys)
  const x = (v: number) => 65 + (v - lowX) / (highX - lowX || 1) * 610
  const y = (v: number) => 345 - (v - lowY) / (highY - lowY || 1) * 295
  return <article className="model-chart"><h3>Normal eigenspace + test patches</h3>
    <div className="model-controls">{['X', 'Y'].map((name, index) => <label key={name}>{name} <select value={axes[index]} onChange={(event) => setAxes(axes.map((a, i) => i === index ? Number(event.target.value) : a))}>{Array.from({ length: model.components }, (_, i) => <option key={i} value={i}>PC{i + 1}</option>)}</select></label>)}</div>
    <svg viewBox="0 0 740 405" role="img" aria-label="PCA scatter plot of training, calibration, and test patches">
      {[0, .5, 1].map((f) => <g key={f}><line x1="65" x2="675" y1={345 - 295 * f} y2={345 - 295 * f} stroke="#e0e8e2" /><text x="58" y={349 - 295 * f} textAnchor="end">{number(lowY + (highY - lowY) * f)}</text><text x={65 + 610 * f} y="368" textAnchor="middle">{number(lowX + (highX - lowX) * f)}</text></g>)}
      {points.map((p, index) => <circle key={index} cx={x(p.scores[axes[0]])} cy={y(p.scores[axes[1]])} r={p.group.startsWith('Test') ? 4 : 2.3} fill={p.color} opacity={p.group.startsWith('Test') ? .9 : .45} onMouseEnter={() => setHover(`${p.group} · ${p.image_id} · patch ${p.patch_id} · error ${number(p.error)}`)}><title>{p.group}: {p.image_id}, patch {p.patch_id}, error {number(p.error)}</title></circle>)}
      <text x="370" y="397" textAnchor="middle">PC{axes[0] + 1}</text><text x="65" y="25">PC{axes[1] + 1}</text>
    </svg>
    <p className="feature-note">Green: training good · Amber: calibration good · Blue/red: normal/anomalous test patches.</p>
    <p className="feature-note">{hover || `${training.length.toLocaleString()} / ${model.training_points.length.toLocaleString()} training points displayed; all points are used by the model and available in CSV.`}</p>
    <p className="feature-note">{model.config.model_type === 'one_class_svm' ? 'This is a two-component view of the SVM inputs. The fitted boundary uses all retained components and is not shown here.' : 'This is a two-component view. Distances use all retained components; reconstruction distance measures what lies outside that space.'}</p>
  </article>
}

function ComponentContributions({ model }: { model: PCAModel }) {
  const [component, setComponent] = useState(0)
  const weights = model.component_weights?.[component] ?? []
  const features = model.feature_names.map((name, i) => ({ name, weight: weights[i] ?? 0, contribution: (weights[i] ?? 0) ** 2 }))
  const ranked = [...features].sort((a, b) => b.contribution - a.contribution)
  const groups = ['L', 'a', 'b', 'gx', 'gy', 'magnitude', 'direction', 'strong_edge', 'hog', 'frangi']
  const groupRows = groups.map((name) => ({ name, share: features.filter((f) => f.name.startsWith(`${name}_`)).reduce((sum, f) => sum + f.contribution, 0) }))
  return <section className="model-composition">
    <div className="panel-heading"><h3>What makes up each component?</h3><label className="model-controls">Component <select value={component} onChange={(e) => setComponent(Number(e.target.value))}>{Array.from({ length: model.components }, (_, i) => <option key={i} value={i}>PC{i + 1} · {percent(model.explained_variance_ratio[i])} variance</option>)}</select></label></div>
    <p className="feature-note">PC{component + 1} explains {percent(model.explained_variance_ratio[component])} of total training variance. Each component is a weighted sum of centered, standardized features. Positive and negative weights indicate opposite directions along its axis; the overall sign is arbitrary.</p>
    <div className="model-composition-grid"><div><h4>Feature families · share of component</h4>{groupRows.map((g) => <div className="model-contribution" key={g.name}><span>{g.name}</span><meter min="0" max="1" value={g.share} aria-label={`${g.name} share`} /><strong>{percent(g.share)}</strong></div>)}</div>
      <div><h4>Strongest individual features</h4><div className="model-table-scroll"><table><thead><tr><th>Feature</th><th>Signed weight</th><th>Component share</th></tr></thead><tbody>{ranked.slice(0, 12).map((f) => <tr key={f.name}><th>{f.name}</th><td>{number(f.weight)}</td><td>{percent(f.contribution)}</td></tr>)}</tbody></table></div></div></div>
    <p className="feature-note">Shares are squared component weights and sum to 100% across all features. They describe component composition, not causal importance or defect detection accuracy. Components explain variance; correlated features do not have a unique independent share of total variance.</p>
    <details className="model-all-weights"><summary>All {features.length} feature weights for PC{component + 1}</summary><div className="model-table-scroll"><table><thead><tr><th>Feature</th><th>Weight</th><th>Component share</th></tr></thead><tbody>{ranked.map((f) => <tr key={f.name}><th>{f.name}</th><td>{number(f.weight)}</td><td>{percent(f.contribution)}</td></tr>)}</tbody></table></div></details>
  </section>
}


export default function Models() {
  const [patchProjection, setPatchProjection] = useState<{ key: string; data?: PCAProjection; error?: string } | null>(null)
  const [allPatches, setAllPatches] = useState(false)
  const [reports, setReports] = useState<PCAReport[]>([])
  const [family, setFamily] = useState('all')
  const [images, setImages] = useState<string[]>([])
  const [imageChoice, setImageChoice] = useState('')
  const [graphChoice, setGraphChoice] = useState('')
  const [job, setJob] = useState<ModelJob | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [preprocessing, setPreprocessing] = useState<AnomalyPreprocessing>('full')
  const [pipelineFilter, setPipelineFilter] = useState('all')
  const [count, setCount] = useState(135)
  useEffect(() => {
    let active = true
    getModels().then((data) => { if (active) { setReports(data.models); setCount(data.retrain_configurations); if (data.active_job) setJob(data.active_job) } }).catch((e: unknown) => { if (active) setError(e instanceof Error ? e.message : 'Could not load models') })
    getImages('metal_plate', 'test').then((data) => { if (active) setImages(data) }).catch((e: unknown) => { if (active) setError(e instanceof Error ? e.message : 'Could not load images') })
    return () => { active = false }
  }, [refresh])
  const jobId = job?.job_id
  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer: number | undefined
    let lastListRefresh = 0
    const poll = async () => {
      try {
        const value = await getModelJob(jobId, controller.signal)
        if (controller.signal.aborted) return
        if (value.status === 'complete' || value.status === 'failed') {
          setJob(null); setRefresh((r) => r + 1)
          if (value.error) setError(value.error)
          if (value.result && 'failures' in value.result && value.result.failures.length) setError(value.result.failures.map((f) => f.error).join('; '))
        } else {
          setJob(value)
          if (Date.now() - lastListRefresh >= 5000) {
            const data = await getModels()
            if (controller.signal.aborted) return
            setReports(data.models); setCount(data.retrain_configurations)
            lastListRefresh = Date.now()
          }
          timer = window.setTimeout(poll, 1000)
        }
      } catch (e) { if (!controller.signal.aborted) { setJob(null); setRefresh((r) => r + 1); setError(e instanceof Error ? e.message : 'Job unavailable') } }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [jobId])
  const retrain = async () => {
    setStarting(true); setError('')
    try { const result = await startPCAJob('retrain_all', { preprocessing }); setJob({ job_id: result.job_id, kind: 'retrain_all', status: 'queued', phase: 'Queued', done: 0, total: 0 }) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not retrain') }
    finally { setStarting(false) }
  }
  const selected = images.includes(imageChoice) ? imageChoice : images[0] ?? ''
  const filtered = reports.filter((r) => (family === 'all' || (r.config.model_type === 'one_class_svm' ? 'svm' : 'pca') === family) && (pipelineFilter === 'all' || (r.config.preprocessing ?? 'legacy') === pipelineFilter))
  const graph = filtered.find((r) => r.model_id === graphChoice) ?? filtered[0]
  const patchState = patchProjection?.key === `${graph?.model_id}:${selected}` ? patchProjection : null
  const patchData = (graph?.compatible !== false || graph?.results_saved) ? patchState?.data : undefined
  const patchMessage = !graph ? 'Select a trained anomaly model to see patch scores.'
    : graph.compatible === false && !graph.results_saved ? 'Patch overlay unavailable: this model was trained with older preprocessing/features. Use Retrain all models to generate compatible patch scores. Saved evaluations contain counts only, so their patch locations cannot be recovered.'
    : patchState?.error ? `Patch overlay unavailable: ${patchState.error}`
    : !patchData ? 'Scoring image patches…'
    : `${patchData.anomalous_patches} of ${patchData.patches.length} patches exceed this model’s patch threshold. ${patchData.anomalous_patches === 0 ? 'No patches were flagged; choose All scored patches to inspect the grid.' : 'Red boxes are flagged patches.'} The image classification uses a separate image threshold.`

  const summary = useModelSort(filtered, 'performance', selected)
  const predictions = useModelSort(filtered, 'correct', selected)
  const classes = [...new Set(filtered.flatMap((r) => Object.keys(r.evaluation?.groups ?? {})))].sort((a, b) => a === b ? 0 : a === 'good' ? -1 : b === 'good' ? 1 : a.localeCompare(b))
  const imageIndex = images.indexOf(selected)
  const moveImage = (delta: number) => { if (images[imageIndex + delta]) setImageChoice(images[imageIndex + delta]) }
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      if (e.target instanceof Element && e.target.closest('input, select, textarea, [contenteditable="true"], .pca-analysis')) return
      if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return
      const next = images[imageIndex + (e.key === 'ArrowLeft' ? -1 : 1)]
      e.preventDefault(); if (next) setImageChoice(next)
    }
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler)
  }, [images, imageIndex])
  return <section className="models-comparison-page"><div className="page-intro model-writeup-intro"><h2>Anomaly detection</h2><AnomalyWriteup /></div>
    <div className="model-controls panel model-training"><label>Training pipeline <select disabled={!!job || starting} value={preprocessing} onChange={e => setPreprocessing(e.target.value as AnomalyPreprocessing)}>{(['full', 'normalized', 'raw'] as const).map(p => <option key={p} value={p}>{pipelineNames[p]}</option>)}</select></label><button disabled={!!job || starting} onClick={() => void retrain()}>Retrain all models ({count})</button><span>All parameter combinations for the selected pipeline. Other pipelines are retained for comparison · CSVs and plot data are kept in CV/results/data. Save selected fitted models to CV/results/models before deleting artifacts.</span><p className="feature-note">Training all anomaly models can take a while because it runs every parameter combination. Keep the server running and your computer awake until it finishes.</p></div>
    <nav className="model-tabs" aria-label="Filter model types">{[['all', 'All models'], ['pca', 'PCA'], ['svm', 'One-class SVM']].map(([value, label]) => <button key={value} aria-current={family === value ? 'page' : undefined} onClick={() => setFamily(value)}>{label}</button>)}</nav>
    {error && <p role="alert" className="panel empty">{error}</p>}
    {job && <div className="panel model-progress" role="status">{job.combinations ? `${job.combination ?? 0} / ${job.combinations} processed · ${job.completed ?? 0} saved · ${job.failed ?? 0} failed · ` : ''}{job.phase} {job.total ? `${job.done} / ${job.total}` : ''}</div>}
    <section className="panel"><div className="panel-heading"><h3>Performance summary</h3><label>Compare pipeline <select value={pipelineFilter} onChange={e => setPipelineFilter(e.target.value)}><option value="all">All pipelines</option>{Object.entries(pipelineNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><span>{filtered.length} models</span></div><p className="feature-note">Default ranking: defect recall, then fewer false alarms. Click headers to sort. Class columns show correctly labeled images as correct/total (percentage): GOOD for good plates, BAD for each defect type.</p>
      <div className="model-table-scroll model-summary-table"><table><thead><tr>{[['name','Model'],['images','Images / skipped'],['accuracy','Accuracy'],['recall','Recall'],['precision','Precision'],['f1','F1'],['fpr','False alarms'],['tp','TP'],['fp','FP'],['tn','TN'],['fn','FN']].map(([key,label]) => <SortHeader key={key} column={key} sort={summary.sort} onSort={summary.choose}>{label}</SortHeader>)}{classes.map((label) => <SortHeader key={label} column={`class:${label}`} sort={summary.sort} onSort={summary.choose}>{label}</SortHeader>)}<th>Keep fitted model</th></tr></thead>
        <tbody>{summary.rows.map((r) => { const e = r.evaluation; return <tr key={r.model_id}><th>{runName(r)}{r.config.feature_set !== 'lab_sobel_hog_frangi' && ' · legacy features'}</th><td>{e ? `${e.images} / ${e.skipped.length}` : 'Not evaluated'}</td><td>{percent(e?.images ? (e.tp + e.tn) / e.images : null)}</td><td>{percent(e?.recall)}</td><td>{percent(e?.precision)}</td><td>{percent(e && 2*e.tp+e.fp+e.fn ? 2*e.tp/(2*e.tp+e.fp+e.fn) : null)}</td><td>{percent(e?.false_positive_rate)}</td>{(['tp','fp','tn','fn'] as const).map((k) => <td key={k}>{e?.[k] ?? '—'}</td>)}{classes.map((label) => { const rows = e?.rows.filter((p) => p.label === label) ?? []; const correct = rows.filter((p) => p.prediction === p.actual).length; return <td key={label}>{rows.length ? `${correct}/${rows.length} (${percent(correct / rows.length)})` : '—'}</td> })}<td><SaveModelButton family="anomaly" modelId={r.model_id} saved={r.model_saved} available={r.model_available} /></td></tr> })}</tbody></table></div>
      {!filtered.length && <p className="empty">No fitted models in this view. Use Retrain all models to populate the comparison.</p>}
    </section>
    <section className="panel"><div className="panel-heading"><h3>Individual predictions</h3><div className="model-controls"><button disabled={imageIndex<=0} onClick={() => moveImage(-1)}>← Image</button><label>Image <select value={selected} onChange={(e) => setImageChoice(e.target.value)}>{images.map((name) => <option key={name}>{name}</option>)}</select></label><button disabled={imageIndex<0 || imageIndex>=images.length-1} onClick={() => moveImage(1)}>Image →</button></div></div>
      <div className="model-prediction-columns"><figure>{selected && <PatchOverlay key={`${graph?.model_id}:${selected}`} src={getImageUrl('metal_plate','test',selected)} alt={selected} patches={patchData ? patchData.patches.filter(p => allPatches || p.anomalous).map(p => ({ ...p, color: p.anomalous ? '#f23b35' : '#3685ce', label: `${p.anomalous ? 'Flagged' : 'Not flagged'} · patch score ${number(p.error)} · threshold ${number(patchData.patch_threshold)}` })) : []} description={patchMessage} />}<figcaption>{selected}</figcaption><div className="view-switch"><button disabled={!patchData} aria-pressed={!allPatches} onClick={() => setAllPatches(false)}>Flagged patches</button><button disabled={!patchData} aria-pressed={allPatches} onClick={() => setAllPatches(true)}>All scored patches</button></div></figure>
        <div className="model-table-scroll prediction-table"><table><thead><tr>{[['name','Model'],['correct','Correct?'],['prediction','Prediction'],['score','Score'],['threshold','Threshold']].map(([key,label]) => <SortHeader key={key} column={key} sort={predictions.sort} onSort={predictions.choose}>{label}</SortHeader>)}</tr></thead><tbody>{predictions.rows.map((r) => { const p=r.evaluation?.rows.find((p)=>p.image_id===`metal_plate/test/${selected}`); return <tr key={r.model_id} aria-selected={graph?.model_id===r.model_id}><th><button onClick={() => setGraphChoice(r.model_id)}>{runName(r)}</button></th><td>{p ? p.prediction===p.actual ? 'Yes' : 'No' : '—'}</td><td>{p ? `${p.prediction} (actual ${p.actual})` : 'Not evaluated'}</td><td>{p ? number(p.score) : '—'}</td><td>{number(r.plate_threshold)}</td></tr> })}</tbody></table></div>
        <div className="prediction-graph"><Projection report={graph} selected={selected} onProjection={setPatchProjection} /></div>
      </div><p className="feature-note">Click a model name to project this image into its PCA space. Correct predictions appear first; unevaluated images have no stored classification.</p>
    </section>
    <PCAAnalysis reports={filtered} />
  </section>
}
