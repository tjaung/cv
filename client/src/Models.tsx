import { useEffect, useState } from 'react'
import { getModels, getPCAModel, getModelJob, startPCAJob, getImages, getImageUrl, getPCAFeaturePage, getPCADownloadUrl } from './api'
import type { ModelJob, PCAModel, PCAReport, PCATest, PCAFeaturePage, PCAEvaluation } from './api'

const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const metricName = (r: PCAReport) => r.config.distance_metric ?? 'squared_l2'
const runName = (r: PCAReport) => `${r.config.patch_size}px · ${percent(r.config.variance_target)} · ${metricName(r)} · ${r.model_id.slice(0, 6)}`
const number = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 3 })

function PCAScatter({ model, test }: { model: PCAModel; test: PCATest | null }) {
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
    <p className="feature-note">This is a two-component view. Distances use all retained components; reconstruction distance measures what lies outside that space.</p>
  </article>
}

function PCADiagnostics({ model, test }: { model: PCAModel; test: PCATest | null }) {
  const variance = model.explained_variance_ratio
  const scores = [model.training_errors, model.calibration_errors, test?.patches.map((p) => p.error) ?? []]
  const colors = ['#278568', '#d99722', '#366ac7']
  const maxLog = Math.max(1, ...scores.flat().map((v) => Math.log10(1 + v)), Math.log10(1 + model.patch_threshold))
  const hist = scores.map((values) => {
    const bins = Array.from({ length: 32 }, () => 0)
    values.forEach((v) => { bins[Math.min(31, Math.floor(Math.log10(1 + v) / maxLog * 32))] += 1 / Math.max(values.length, 1) })
    return bins
  })
  const maxHeight = Math.max(.01, ...hist.flat())
  return <div className="model-diagnostics">
    <article className="model-chart"><h3>PCA explained variance</h3><svg viewBox="0 0 620 280" role="img" aria-label="PCA scree and cumulative explained variance">
      {variance.map((v, i) => { const cumulative = variance.slice(0, i + 1).reduce((sum, value) => sum + value, 0); return <g key={i}><rect x={45 + i / variance.length * 550} y={220 - v * 180} width={Math.max(1, 550 / variance.length - 1)} height={v * 180} fill={i < model.components ? '#278568' : '#bccdc3'}><title>PC{i + 1}: {percent(v)} variance</title></rect><circle cx={45 + (i + .5) / variance.length * 550} cy={220 - cumulative * 180} r="2" fill="#486bd4"><title>Cumulative through PC{i + 1}: {percent(cumulative)}</title></circle></g> })}
      {[0, .5, 1].map((v) => <text key={v} x="37" y={224 - v * 180} textAnchor="end">{Math.round(v * 100)}%</text>)}
      <text x="45" y="247">PC1</text><text x="590" y="247" textAnchor="end">PC{variance.length}</text><text x="310" y="269" textAnchor="middle">Bars: individual variance · Blue: cumulative</text>
    </svg><p className="feature-note">{model.components} components retain {percent(model.retained_variance)} of training variance. This target is separate from the 99th-percentile error thresholds.</p></article>
    <article className="model-chart"><h3>Patch anomaly scores</h3><svg viewBox="0 0 620 280" role="img" aria-label="Anomaly score distributions and calibration threshold">
      {hist.map((values, i) => <path key={i} d={values.map((v, j) => `${j ? 'L' : 'M'}${45 + (j + .5) / 32 * 550},${220 - v / maxHeight * 180}`).join(' ')} fill="none" stroke={colors[i]} strokeWidth="2" />)}
      <line x1={45 + Math.log10(1 + model.patch_threshold) / maxLog * 550} x2={45 + Math.log10(1 + model.patch_threshold) / maxLog * 550} y1="30" y2="220" stroke="#c4373d" strokeDasharray="5 4" />
      {[0, .5, 1].map((f) => <text key={f} x={45 + f * 550} y="245" textAnchor="middle">{number(10 ** (maxLog * f) - 1)}</text>)}
      <text x="310" y="269" textAnchor="middle">Score · log(1 + score) spacing</text><text x="45" y="20">Relative frequency · peak {percent(maxHeight)}</text>
    </svg><p className="feature-note">Green: training · Amber: held-out good · Blue: current test. Dashed line: 99th-percentile patch threshold ({number(model.patch_threshold)}).</p></article>
  </div>
}

function ComponentContributions({ model }: { model: PCAModel }) {
  const [component, setComponent] = useState(0)
  const weights = model.component_weights?.[component] ?? []
  const features = model.feature_names.map((name, i) => ({ name, weight: weights[i] ?? 0, contribution: (weights[i] ?? 0) ** 2 }))
  const ranked = [...features].sort((a, b) => b.contribution - a.contribution)
  const groups = ['L', 'a', 'b', 'gx', 'gy', 'magnitude', 'direction', 'strong_edge']
  const groupRows = groups.map((name) => ({ name, share: features.filter((f) => f.name.startsWith(`${name}_`)).reduce((sum, f) => sum + f.contribution, 0) }))
  return <section className="panel model-composition">
    <div className="panel-heading"><h3>What makes up each component?</h3><label className="model-controls">Component <select value={component} onChange={(e) => setComponent(Number(e.target.value))}>{Array.from({ length: model.components }, (_, i) => <option key={i} value={i}>PC{i + 1} · {percent(model.explained_variance_ratio[i])} variance</option>)}</select></label></div>
    <p className="feature-note">PC{component + 1} explains {percent(model.explained_variance_ratio[component])} of total training variance. Each component is a weighted sum of centered, standardized features. Positive and negative weights indicate opposite directions along its axis; the overall sign is arbitrary.</p>
    <div className="model-composition-grid"><div><h4>Feature families · share of component</h4>{groupRows.map((g) => <div className="model-contribution" key={g.name}><span>{g.name}</span><meter min="0" max="1" value={g.share} aria-label={`${g.name} share`} /><strong>{percent(g.share)}</strong></div>)}</div>
      <div><h4>Strongest individual features</h4><div className="model-table-scroll"><table><thead><tr><th>Feature</th><th>Signed weight</th><th>Component share</th></tr></thead><tbody>{ranked.slice(0, 12).map((f) => <tr key={f.name}><th>{f.name}</th><td>{number(f.weight)}</td><td>{percent(f.contribution)}</td></tr>)}</tbody></table></div></div></div>
    <p className="feature-note">Shares are squared component weights and sum to 100% across all features. They describe component composition, not causal importance or defect detection accuracy. Components explain variance; correlated features do not have a unique independent share of total variance.</p>
    <details className="model-all-weights"><summary>All {features.length} feature weights for PC{component + 1}</summary><div className="model-table-scroll"><table><thead><tr><th>Feature</th><th>Weight</th><th>Component share</th></tr></thead><tbody>{ranked.map((f) => <tr key={f.name}><th>{f.name}</th><td>{number(f.weight)}</td><td>{percent(f.contribution)}</td></tr>)}</tbody></table></div></details>
  </section>
}

function FeatureTable({ modelId, test }: { modelId: string; test: PCATest | null }) {
  const [split, setSplit] = useState('training')
  const [view, setView] = useState('raw')
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<PCAFeaturePage | null>(null)
  const [error, setError] = useState('')
  const [loadedKey, setLoadedKey] = useState('')
  const key = `${split}:${view}:${offset}:${test?.test_id}`
  useEffect(() => {
    const controller = new AbortController()
    getPCAFeaturePage(modelId, split, view, offset, test?.test_id, controller.signal).then((value) => { setPage(value); setLoadedKey(key); setError('') }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load features')
    })
    return () => controller.abort()
  }, [modelId, split, view, offset, test?.test_id, key])
  return <section className="panel model-data"><div className="panel-heading"><h3>Complete feature data</h3><div className="model-controls">
    <label>Samples <select value={split} onChange={(e) => { setSplit(e.target.value); setOffset(0) }}><option value="training">Training good</option><option value="calibration">Held-out good</option>{test && <option value="test">Current test</option>}</select></label>
    <label>Values <select value={view} onChange={(e) => { setView(e.target.value); setOffset(0) }}><option value="raw">Raw features</option><option value="standardized">Standardized</option><option value="reconstructed">Reconstructed (standardized)</option><option value="pca">PCA coordinates</option></select></label>
    <a href={getPCADownloadUrl(modelId, split, test?.test_id)}>Download complete CSV</a><a href={getPCADownloadUrl(modelId, 'components')}>PCA loadings CSV</a>
    </div></div><p className="feature-note">Every patch and feature is available here. CSV includes raw values, standardized values, standardized reconstructions, errors, and every retained PCA coordinate. Scroll horizontally for all columns.</p>
    {error ? <p role="alert" className="feature-note">{error}</p> : loadedKey !== key || !page ? <p className="empty">Loading feature rows…</p> : <>
      <div className="model-table-scroll"><table><thead><tr><th>Image</th><th>Patch</th><th>Bounds</th><th>Coverage</th><th>Error</th>{page.columns.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{page.rows.map((row, i) => <tr key={i}><td>{row.record.image_id}</td><td>{row.record.patch_id}</td><td>{row.record.left},{row.record.top}–{row.record.right},{row.record.bottom}</td><td>{percent(row.record.coverage)}</td><td>{number(row.error)}</td>{row.values.map((value, j) => <td key={j}>{number(value)}</td>)}</tr>)}</tbody></table></div>
      <div className="model-controls feature-note"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 15))}>← Previous rows</button><span>{offset + 1}–{Math.min(offset + 15, page.total)} / {page.total} patches · {page.columns.length} columns</span><button disabled={offset + 15 >= page.total} onClick={() => setOffset(offset + 15)}>Next rows →</button></div>
    </>}
  </section>
}

function EvaluatedImages({ evaluation, selected, onSelect, threshold, busy }: {
  evaluation: PCAEvaluation; selected: string; onSelect: (image: string) => void; threshold: number; busy: boolean
}) {
  const rows = evaluation.rows
  const index = rows.findIndex((row) => row.image_id === `metal_plate/test/${selected}`)
  const row = rows[index]
  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (busy || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      if (event.target instanceof Element && event.target.closest('input, select, textarea, [contenteditable]:not([contenteditable="false"]), [role="slider"], [role="textbox"]')) return
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      event.preventDefault()
      const next = index < 0 ? 0 : index + (event.key === 'ArrowLeft' ? -1 : 1)
      if (rows[next]) onSelect(rows[next].image_id.replace(/^metal_plate\/test\//, ''))
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [busy, index, rows, onSelect])
  if (!rows.length) return null
  return <section className="panel model-evaluated">
    <div className="panel-heading"><h3>Evaluated images</h3><div className="model-controls">
      <button disabled={busy || index <= 0} onClick={() => onSelect(rows[index - 1].image_id.replace(/^metal_plate\/test\//, ''))}>← Previous image</button>
      <span>{index < 0 ? 'No evaluated image selected' : `${index + 1} / ${rows.length}`}</span>
      <button disabled={busy || index >= rows.length - 1} onClick={() => onSelect(rows[index + 1].image_id.replace(/^metal_plate\/test\//, ''))}>Next image →</button>
    </div></div>
    <p className="feature-note">Use ← / → to browse saved evaluation results. Select “Score image” for the current image’s full PCA details and anomaly map.</p>
    {row ? <div className="model-evaluated-content" key={row.image_id}>
      <img src={getImageUrl('metal_plate', 'test', selected)} alt={`Evaluated plate ${selected}`} />
      <div aria-live="polite" aria-atomic="true"><h3>{selected.split('/').at(-1)}</h3>
        <p>Prediction: <strong className={row.prediction === 'BAD' ? 'model-bad' : 'model-good'}>{row.prediction}</strong></p>
        <p>Actual: <strong>{row.actual}</strong> · {row.label}</p>
        <p>Plate score: <strong>{number(row.score)}</strong></p><p>Threshold: {number(threshold)}</p>
        <p>{row.anomalous_patches} / {row.patches} anomalous patches</p>
        <p>{row.prediction === row.actual ? 'Correct classification' : 'Misclassified'}</p>
      </div>
    </div> : <p className="feature-note">This image has no saved evaluation result. Use the navigation buttons to select an evaluated image.</p>}
  </section>
}

function TestResults({ test }: { test: PCATest }) {
  const [selected, setSelected] = useState(0)
  const patch = test.patches[selected]
  return <section className="panel"><div className="panel-heading"><h3>{test.image_id.split('/').at(-1)} · <span className={test.prediction === 'BAD' ? 'model-bad' : 'model-good'}>{test.prediction}</span></h3><span>Plate score {number(test.plate_score)} / threshold {number(test.plate_threshold)}</span></div>
    <p className="feature-note">{test.anomalous_patches} / {test.patches.length} patches exceed the patch threshold. The plate verdict uses its separately calibrated maximum-error threshold. {test.membership !== 'unseen' && `This image belongs to ${test.membership}; this is not an independent test.`}</p>
    <div className="model-image-grid"><figure><img src={test.image} alt="Test plate with anomalous patch rectangles" /><figcaption>Red boxes: patches exceeding the 99th-percentile patch threshold</figcaption></figure><figure><img src={test.anomaly_map} alt="Patch reconstruction-error anomaly map" /><figcaption>Error map: 0 (dark) → {number(test.map_max)} or more (bright)</figcaption></figure><figure><img src={test.coverage_mask} alt="Evaluated plate patch coverage" /><figcaption>White: evaluated coverage; excluded regions have no score</figcaption></figure></div>
    <div className="model-controls feature-note"><label>Inspect patch <select value={selected} onChange={(e) => setSelected(Number(e.target.value))}>{test.patches.map((p, i) => <option key={i} value={i}>#{p.patch_id} · error {number(p.error)}{p.anomalous ? ' · anomalous' : ''}</option>)}</select></label><a href={getPCADownloadUrl(test.model_id, 'test', test.test_id)}>Download test vectors</a></div>
    {patch && <p className="feature-note">Patch #{patch.patch_id} · ({patch.left}, {patch.top})–({patch.right}, {patch.bottom}) · nearest training Euclidean distance: <strong>{number(patch.nearest_distance)}</strong> · variance-weighted PCA distance: <strong>{number(patch.mahalanobis)}</strong> · nearest good reference: {patch.nearest_patch.image_id}, patch #{patch.nearest_patch.patch_id}. These distances are diagnostics; the decision uses the selected reconstruction distance.</p>}
  </section>
}

function HolisticModels({ reports, selected, images, onImage, onModel, busy, onEvaluate }: {
  reports: PCAReport[]; selected: string; images: string[]; onImage: (value: string) => void; onModel: (value: string) => void; busy: boolean; onEvaluate: () => void
}) {
  return <section className="panel"><div className="panel-heading"><h3>All experiments · test results & PCA summaries</h3><button disabled={busy || !reports.length} onClick={onEvaluate}>Evaluate all compatible models</button></div>
    <p className="feature-note">Every saved run is listed. Comparisons use the same test set; missing or skipped images are not counted as correct. Test results are for comparison, not threshold calibration.</p>
    <div className="model-table-scroll"><table><thead><tr><th>Run · parameters</th><th>Train / held out</th><th>Features → PCs</th><th>Retained variance</th><th>Images / skipped</th><th>Accuracy</th><th>Recall</th><th>Precision</th><th>F1</th><th>False alarms</th><th>TP / FP / TN / FN</th></tr></thead><tbody>{reports.map((r) => { const e = r.evaluation; return <tr key={r.model_id}><th><button disabled={busy} onClick={() => onModel(r.model_id)}>{runName(r)}</button>{r.compatible === false && ' · old pipeline'}</th><td>{r.training_images} / {r.calibration_images}</td><td>{r.features} → {r.components}</td><td>{percent(r.retained_variance)}</td><td>{e ? `${e.images} / ${e.skipped.length}` : 'Not evaluated'}</td><td>{percent(e?.images ? (e.tp + e.tn) / e.images : null)}</td><td>{percent(e?.recall)}</td><td>{percent(e?.precision)}</td><td>{percent(e && 2 * e.tp + e.fp + e.fn ? 2 * e.tp / (2 * e.tp + e.fp + e.fn) : null)}</td><td>{percent(e?.false_positive_rate)}</td><td>{e ? `${e.tp} / ${e.fp} / ${e.tn} / ${e.fn}` : '—'}</td></tr> })}</tbody></table></div>
    {!reports.length && <p className="empty">Train an experiment above to start comparing models.</p>}
    <details className="model-all-weights"><summary>PCA variance profiles · all {reports.length} runs</summary><div className="model-table-scroll"><table><thead><tr><th>Run</th><th>Variance by component (first 12)</th><th>Cumulative retained variance</th><th>Loadings</th></tr></thead><tbody>{reports.map((r) => <tr key={r.model_id}><th>{runName(r)}</th><td>{r.explained_variance_ratio.slice(0, 12).map((v, i) => <span className="model-pc-share" key={i}>PC{i + 1}: {percent(v)}{i >= r.components ? ' (excluded)' : ''} </span>)}</td><td>{percent(r.retained_variance)} across {r.components} PCs</td><td><a href={getPCADownloadUrl(r.model_id, 'components')}>All component weights CSV</a></td></tr>)}</tbody></table></div><p className="feature-note">Select a run below to inspect its full variance plot and every component’s feature weights. Distance metrics change anomaly scoring; runs with the same patches, split, and variance target learn the same PCA basis.</p></details>
    <div className="panel-heading"><h3>Image comparison matrix</h3><label className="model-controls">Image <select value={selected} disabled={busy} onChange={(e) => onImage(e.target.value)}>{images.map((name) => <option key={name}>{name}</option>)}</select></label></div>
    <div className="model-matrix"><div>{selected && <img src={getImageUrl('metal_plate', 'test', selected)} alt={`Comparison ${selected}`} />}</div><div className="model-table-scroll"><table><thead><tr><th>Model</th><th>Actual</th><th>Prediction</th><th>Score</th><th>Threshold</th><th>Score / threshold</th><th>Anomalous patches</th></tr></thead><tbody>{reports.map((r) => { const row = r.evaluation?.rows.find((p) => p.image_id === `metal_plate/test/${selected}`); return <tr key={r.model_id}><th><button disabled={busy} onClick={() => onModel(r.model_id)}>{runName(r)}</button></th><td>{row?.actual ?? '—'}</td><td className={row?.prediction === 'BAD' ? 'model-bad' : 'model-good'}>{row?.prediction ?? (r.evaluation?.skipped.some((p) => p.image_id === `metal_plate/test/${selected}`) ? 'Unscorable' : 'Not evaluated')}</td><td>{row ? number(row.score) : '—'}</td><td>{number(r.plate_threshold)}</td><td>{row && r.plate_threshold > 0 ? number(row.score / r.plate_threshold) : '—'}</td><td>{row ? `${row.anomalous_patches} / ${row.patches}` : '—'}</td></tr> })}</tbody></table></div></div>
    <p className="feature-note">Score / threshold above 1 means BAD. This ratio compares each run with its own cutoff; it is not a probability. Raw scores use different units across distance metrics.</p>
  </section>
}

function ModelSelector({ reports, value, onChange, busy }: { reports: PCAReport[]; value: string; onChange: (value: string) => void; busy: boolean }) {
  const [patch, setPatch] = useState('all'), [variance, setVariance] = useState('all'), [metric, setMetric] = useState('all')
  const matches = reports.filter((r) => (patch === 'all' || String(r.config.patch_size) === patch) && (variance === 'all' || String(r.config.variance_target) === variance) && (metric === 'all' || metricName(r) === metric))
  return <section className="panel model-training"><h3>Inspect one model</h3><div className="model-controls">
    <label>Patch size <select disabled={busy} value={patch} onChange={(e) => setPatch(e.target.value)}><option value="all">All</option>{[...new Set(reports.map((r) => r.config.patch_size))].sort((a,b) => a-b).map((v) => <option key={v}>{v}</option>)}</select></label>
    <label>Variance target <select disabled={busy} value={variance} onChange={(e) => setVariance(e.target.value)}><option value="all">All</option>{[...new Set(reports.map((r) => r.config.variance_target))].sort().map((v) => <option key={v} value={v}>{percent(v)}</option>)}</select></label>
    <label>Distance <select disabled={busy} value={metric} onChange={(e) => setMetric(e.target.value)}><option value="all">All</option>{[...new Set(reports.map(metricName))].sort().map((v) => <option key={v}>{v}</option>)}</select></label>
    <label>Saved run <select disabled={busy} value={matches.some((r) => r.model_id === value) ? value : ''} onChange={(e) => onChange(e.target.value)}><option value="" disabled>Choose matching run</option>{matches.map((r) => <option key={r.model_id} value={r.model_id}>{runName(r)}</option>)}</select></label>
    </div><p>{matches.length} matching runs. Inspecting: {reports.find((r) => r.model_id === value) ? runName(reports.find((r) => r.model_id === value)!) : 'None'}</p>
  </section>
}

export default function Models() {
  const [reports, setReports] = useState<PCAReport[]>([])
  const [modelId, setModelId] = useState('')
  const [distanceMetric, setDistanceMetric] = useState('all')
  const [model, setModel] = useState<PCAModel | null>(null)
  const [test, setTest] = useState<PCATest | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<ModelJob | null>(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)
  const [images, setImages] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [patchSize, setPatchSize] = useState(64)
  const [variance, setVariance] = useState(.95)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let active = true
    getModels().then(async (data) => {
      if (!active) return
      setReports(data.models)
      if (data.active_job) { setJobId(data.active_job.job_id); setJob(data.active_job) }
      setModelId((current) => data.models.some((r) => r.model_id === current) ? current : data.models[0]?.model_id ?? '')
    }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Could not load models') })
    getImages('metal_plate', 'test').then((names) => { if (active) { setImages(names); setSelected((current) => current || names[0] || '') } }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Could not load test images') })
    return () => { active = false }
  }, [refresh])
  useEffect(() => {
    if (!modelId) return
    let active = true
    getPCAModel(modelId).then((fitted) => {
      if (active) { setModel(fitted); setTest((current) => current?.model_id === fitted.model_id ? current : null) }
    }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Could not load model') })
    return () => { active = false }
  }, [modelId, refresh])
  useEffect(() => {
    if (!jobId) return
    const controller = new AbortController()
    let timer: number | undefined
    const poll = async () => {
      try {
        const value = await getModelJob(jobId, controller.signal)
        if (controller.signal.aborted) return
        setJob(value)
        if (value.status === 'complete') {
          if (value.kind === 'test') setTest(value.result as PCATest)
          if (value.kind === 'train') setTest(null)
          if (value.result && 'failures' in value.result && value.result.failures.length) setError(value.result.failures.map((f) => f.error).join('; '))
          setJobId(null); setRefresh((current) => current + 1)
        } else if (value.status === 'failed') { setError(value.error ?? 'Model job failed'); setJobId(null) }
        else timer = window.setTimeout(poll, 1000)
      } catch (reason) { if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : 'Job status unavailable'); setJobId(null) } }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [jobId])
  const start = async (kind: ModelJob['kind']) => {
    setError(''); setStarting(true)
    try {
      const body = kind === 'sweep' ? { patch_sizes: patchSize ? [patchSize] : [32, 64, 128], variance_targets: variance ? [variance] : [.9, .95, .99], distance_metrics: distanceMetric === 'all' ? ['l1', 'l2', 'mahalanobis'] : [distanceMetric] } : kind === 'train' ? { patch_size: patchSize, variance_target: variance, distance_metric: distanceMetric } : kind === 'test' ? { image_path: `metal_plate/test/${selected}` } : {}
      const started = await startPCAJob(kind, body, model?.model_id)
      setJobId(started.job_id); setJob({ job_id: started.job_id, kind, status: 'queued', phase: 'Queued', done: 0, total: 0 })
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not start model job') }
    finally { setStarting(false) }
  }
  const busy = !!jobId || starting || (!!modelId && model?.model_id !== modelId)
  const evaluation = reports.find((r) => r.model_id === modelId)?.evaluation
  const activeTest = test?.model_id === modelId && test?.image_id === `metal_plate/test/${selected}` ? test : null
  return <section>
    <div className="page-intro"><p className="eyebrow">METAL PLATE · ANOMALY DETECTION</p><h2>Models</h2><p>Learn normal patch features from good training images. Hold out entire good images for calibration, then compare new patches with the learned PCA subspace. Only LAB and Sobel features enter this model.</p></div>
    <nav className="model-tabs" aria-label="Model selection"><button aria-current="page">PCA</button></nav>
    <div className="panel model-training"><h3>Train PCA experiments</h3><p>Learn normal LAB and Sobel features from good plates. Calibrate anomaly thresholds on held-out good images at the 99th percentile.</p>
      <div className="model-controls"><label>Patch size <select value={patchSize} disabled={busy} onChange={(e) => setPatchSize(Number(e.target.value))}><option value="0">All patch sizes</option><option value="32">32 × 32</option><option value="64">64 × 64</option><option value="128">128 × 128</option></select></label><label>PCA variance target <select value={variance} disabled={busy} onChange={(e) => setVariance(Number(e.target.value))}><option value="0">All variance targets</option><option value="0.9">90%</option><option value="0.95">95%</option><option value="0.99">99%</option></select></label><label>Distance <select value={distanceMetric} disabled={busy} onChange={(e) => setDistanceMetric(e.target.value)}><option value="all">All distances</option><option value="l1">L1</option><option value="l2">L2</option><option value="mahalanobis">Mahalanobis</option></select></label><button disabled={busy} onClick={() => void start('sweep')}>Train & evaluate {(patchSize ? 1 : 3) * (variance ? 1 : 3) * (distanceMetric === 'all' ? 3 : 1)} combinations</button><span>Error calibration is fixed at the 99th percentile.</span></div>
      <details><summary>Training and threshold details</summary><p>Good images are split approximately 80/20 by image (seed 42); identical files stay together. Test images never fit the scaler, PCA, or thresholds. Features use the normalized segmented plate before glare repair.</p><p>Distances score the standardized reconstruction residual: L1 sums absolute values; L2 takes Euclidean length; Mahalanobis uses training-only residual covariance with a 1% average-variance ridge (minimum 1e-8). Legacy models retain squared L2. Each run calibrates its own threshold; raw scores across metrics have different units. Patches below 50% plate coverage are excluded. The plate score is its maximum patch error, calibrated separately using held-out image maxima; one flagged patch does not automatically make a plate BAD.</p>
      {model && <p>Current run: {model.config.patch_size}px patches · {model.training_patches.toLocaleString()} training patches · {model.calibration_patches.toLocaleString()} calibration patches · patch threshold {number(model.patch_threshold)} · plate threshold {number(model.plate_threshold)}. The 99th percentile is an empirical cutoff, not 99% confidence; {model.calibration_images} held-out images give limited tail calibration.</p>}</details>
    </div>
    {error && <p className="panel empty" role="alert">{error}</p>}
    {busy && <div className="panel model-progress" role="status"><p>{job?.combinations ? `Combination ${job.combination} / ${job.combinations} · ` : ''}{job?.phase ?? 'Starting…'} {job?.total ? `${job.done} / ${job.total}` : ''}</p>{!!job?.total && <progress value={job.done} max={job.total} />}<p>Jobs run on the server; source images are never overwritten.</p></div>}
    <HolisticModels reports={reports} selected={selected} images={images} onImage={setSelected} onModel={setModelId} busy={busy} onEvaluate={() => void start('evaluate_all')} />
    <ModelSelector reports={reports} value={modelId} onChange={setModelId} busy={!!jobId || starting} />
    {model && model.model_id === modelId && <>
      {evaluation && <section className="panel"><div className="panel-heading"><h3>Test-set breakdown · fixed thresholds</h3><a href={getPCADownloadUrl(model.model_id, 'evaluation')}>Download predictions</a></div><div className="model-table-scroll"><table><thead><tr><th>Class</th><th>Images</th><th>Flagged BAD</th><th>Flag rate</th></tr></thead><tbody>{Object.entries(evaluation.groups).map(([label, group]) => <tr key={label}><th>{label}</th><td>{group.images}</td><td>{group.flagged}</td><td>{percent(group.flagged / group.images)}</td></tr>)}</tbody></table></div><p className="feature-note">TP {evaluation.tp} · FP {evaluation.fp} · TN {evaluation.tn} · FN {evaluation.fn}. These results are evaluation only; thresholds are not adjusted from test labels.</p>{evaluation.skipped.length > 0 && <details className="feature-note"><summary>{evaluation.skipped.length} unscorable images</summary>{evaluation.skipped.map((r) => <p key={r.image_id}>{r.image_id}: {r.reason}</p>)}</details>}</section>}
      {reports.find((r) => r.model_id === modelId)?.compatible === false && <p className="panel empty">This run uses an older preprocessing pipeline. Its saved results remain available; train a new run to score current images.</p>}
      <div className="panel model-controls model-test-controls"><label>Test image <select value={selected} disabled={busy} onChange={(e) => setSelected(e.target.value)}>{images.map((name) => <option key={name}>{name}</option>)}</select></label><button disabled={busy || !selected || reports.find((r) => r.model_id === modelId)?.compatible === false} onClick={() => void start('test')}>Score image</button><button disabled={busy || reports.find((r) => r.model_id === modelId)?.compatible === false} onClick={() => void start('evaluate')}>Evaluate all test images</button></div>
      <div className="model-inspection"><div>
      {evaluation && <EvaluatedImages evaluation={evaluation} selected={selected} onSelect={setSelected} threshold={model.plate_threshold} busy={busy} />}
      {!evaluation && selected && <section className="panel model-preview"><h3>{selected}</h3><img src={getImageUrl('metal_plate', 'test', selected)} alt={selected} /><p>Score this image to see its classification and PCA projection.</p></section>}
      </div><section className="panel">{!activeTest && <p className="feature-note">Select “Score image” to project this image’s patches onto the chart. Currently showing normal reference patches only.</p>}<PCAScatter key={model.model_id} model={model} test={activeTest} /></section></div>
      {activeTest && <details className="panel model-result-details"><summary>Current image · {activeTest.prediction} · score {number(activeTest.plate_score)} · anomaly map and patch distances</summary><TestResults key={activeTest.test_id} test={activeTest} /></details>}
      <section className="panel"><div className="panel-heading"><h3>PCA variance & reconstruction</h3><span>{model.components} components · {percent(model.retained_variance)} retained</span></div><PCADiagnostics model={model} test={activeTest} /></section>
      <ComponentContributions key={model.model_id} model={model} />

      {model.skipped.length > 0 && <details className="panel feature-note"><summary>Skipped training images ({model.skipped.length})</summary>{model.skipped.map((r) => <p key={r.image_id}>{r.image_id}: {r.reason}</p>)}</details>}
      <FeatureTable key={`${model.model_id}:${activeTest?.test_id}`} modelId={model.model_id} test={activeTest} />
    </>}
  </section>
}
