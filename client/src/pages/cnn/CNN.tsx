import { CNNWriteup } from './CNNWriteup'
import { useEffect, useState } from 'react'
import { cnnAction, getCNN, getCNNInspection, getCNNPredictions, getCNNWeights, getImageUrl, getModelJob } from '../../api'
import type { CNNInspection, CNNMetrics, CNNPrediction, CNNRun, CNNWeights, ModelJob } from '../../api'

const isPatch = (run?: CNNRun) => run?.config.variant === 'patch' || run?.config.variant === 'patch_multiclass'
const variantName = (run: CNNRun) => run.config.variant === 'patch_multiclass' ? 'Patch four-class' : run.config.variant === 'patch' ? 'Patch good/defect' : run.config.variant === 'defect_weighted' ? 'Defect-weighted' : 'Standard'
const percent = (x: number) => `${(100*x).toFixed(1)}%`
function score(metrics: CNNMetrics, key: string) {
  const row = metrics.report[key]
  return typeof row === 'object' ? row : undefined
}
const architecture = [
  ['Input', '3 × 224 × 224', 'Segmented RGB plate or patch + ImageNet normalization'],
  ['Stem', '64 × 56 × 56', '7×7 convolution → batch norm → ReLU → max pool'],
  ['Residual stage 1', '64 × 56 × 56', '2 basic blocks, identity shortcuts'],
  ['Residual stage 2', '128 × 28 × 28', '2 basic blocks, downsampling shortcut'],
  ['Residual stage 3', '256 × 14 × 14', '2 basic blocks, downsampling shortcut'],
  ['Residual stage 4', '512 × 7 × 7', '2 basic blocks, downsampling shortcut'],
  ['Average pooling', '512 features', 'Spatial average of each learned channel'],
  ['Final linear layer', '4-class or binary logits', '512 features → class scores; softmax for display'],
]

function LearningCurve({ run }: { run: CNNRun }) {
  const max = Math.max(1, ...run.history.map(p => p.loss))
  return <figure><figcaption>{run.config.variant === 'defect_weighted' ? 'Weighted' : 'Standard'} training cross-entropy by epoch</figcaption><svg viewBox="0 0 480 150" role="img" aria-label="Training loss curve">
    <path d="M35 10 V120 H470" fill="none" stroke="#82919a" />
    <text x="0" y="15" fontSize="12">{max.toFixed(2)}</text><text x="15" y="122" fontSize="12">0</text>
    <polyline fill="none" stroke="#237b94" strokeWidth="2" points={run.history.map((p, i) => `${40+i*420/Math.max(1, run.history.length-1)},${120-p.loss/max*100}`).join(' ')} />
    {run.history.map((p, i) => <circle key={p.epoch} cx={40+i*420/Math.max(1, run.history.length-1)} cy={120-p.loss/max*100} r="3" fill="#237b94"><title>Epoch {p.epoch}: {p.loss.toFixed(4)}</title></circle>)}
    <text x="35" y="145" fontSize="12">Epoch 1</text><text x="400" y="145" fontSize="12">Epoch {run.config.epochs}</text>
  </svg></figure>
}

function WeightView({ run, weights }: { run: CNNRun; weights: CNNWeights }) {
  const maximum = Math.max(1e-9, ...weights.classifier_weights.flat().map(Math.abs))
  return <div><div className="cnn-columns"><figure><figcaption>64 learned first-layer RGB filters (7 × 7)</figcaption><img className="cnn-filters" src={weights.conv1_filters} alt="Grid of first convolution filters" /><p>Shared scale ±{weights.conv1_scale.toFixed(4)}. Gray is zero; brighter and darker channels represent positive and negative weights.</p></figure>
    <div><h4>Final classifier weights: 512 channels per class</h4><p>Blue = negative, red = positive, white = zero. These are learned channels, not named LAB or Sobel features. Hover a cell to inspect its weight.</p>
      <svg viewBox="0 0 650 160" role="img" aria-label={`${run.class_names.length} by 512 classifier weight heatmap`}>{weights.classifier_weights.map((row, i) => <g key={i}><text x="0" y={i*35+20} fontSize="11">{run.class_names[i]}</text>{row.map((w, j) => <rect key={j} x={135+j} y={i*35+3} width="1" height="26" fill={w >= 0 ? `rgba(196,55,42,${Math.abs(w)/maximum})` : `rgba(35,104,180,${Math.abs(w)/maximum})`}><title>{run.class_names[i]} · channel {j} · {w.toFixed(5)}</title></rect>)}</g>)}</svg>
      <p>Shared range: −{maximum.toFixed(4)} to +{maximum.toFixed(4)}. Biases: {weights.classifier_bias.map(x => x.toFixed(3)).join(', ')}.</p><LearningCurve run={run} /></div></div>
    <details><summary>Weight statistics for every layer</summary><div className="model-table-scroll"><table><thead><tr><th>Layer</th><th>Parameters</th><th>Mean</th><th>Standard deviation</th></tr></thead><tbody>{weights.layers.map(l => <tr key={l.name}><th>{l.name}</th><td>{l.parameters.toLocaleString()}</td><td>{l.mean.toFixed(5)}</td><td>{l.std.toFixed(5)}</td></tr>)}</tbody></table></div></details>
  </div>
}

export default function CNN() {
  const [runs, setRuns] = useState<CNNRun[]>([])
  const [choice, setChoice] = useState('')
  const [job, setJob] = useState<ModelJob | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [variant, setVariant] = useState<'standard' | 'defect_weighted' | 'patch' | 'patch_multiclass'>('standard')
  const [epochs, setEpochs] = useState(5)
  const [rate, setRate] = useState(.0001)
  const [workers, setWorkers] = useState(2)
  const [batch, setBatch] = useState(16)
  const [revision, setRevision] = useState(0)
  const [data, setData] = useState<{ id: string; weights: CNNWeights; predictions: CNNPrediction[] }>()
  const [index, setIndex] = useState(0)
  const [inspection, setInspection] = useState<{ key: string; value: CNNInspection }>()
  const run = runs.find(r => r.run_id === choice) ?? runs[0]
  const id = run?.run_id ?? ''
  const current = data?.id === id ? data : undefined
  const rows = current?.predictions ?? []
  const selected = rows[index]
  const view = inspection?.key === `${id}:${index}:${revision}` ? inspection.value : undefined

  useEffect(() => {
    const controller = new AbortController()
    getCNN(controller.signal).then(v => { setRuns(v.runs); setJob(v.active_job) }).catch(e => { if (!controller.signal.aborted) setError(String(e)) })
    return () => controller.abort()
  }, [revision])
  useEffect(() => {
    if (!job) return
    let cancelled = false
    const timer = window.setInterval(() => {
      getModelJob(job.job_id).then(value => {
        if (cancelled) return
        if (value.status === 'failed' || value.status === 'complete') {
          if (value.error) setError(value.error)
          setJob(null); setRevision(r => r+1)
        } else setJob(value)
      }).catch(e => { if (!cancelled) setError(String(e)) })
    }, 1200)
    return () => { cancelled = true; clearInterval(timer) }
  }, [job])
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    Promise.all([getCNNWeights(id, controller.signal), getCNNPredictions(id, controller.signal)]).then(([weights, predictions]) => {
      if (!controller.signal.aborted) setData({ id, weights, predictions })
    }).catch(e => { if (!controller.signal.aborted) setError(String(e)) })
    return () => controller.abort()
  }, [id, revision])
  useEffect(() => {
    if (!id || !selected) return
    const controller = new AbortController()
    getCNNInspection(id, index, controller.signal).then(value => { if (!controller.signal.aborted) setInspection({ key: `${id}:${index}:${revision}`, value }) }).catch(e => { if (!controller.signal.aborted) setError(String(e)) })
    return () => controller.abort()
  }, [id, index, selected, revision])
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || (e.target instanceof HTMLElement && (e.target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)))) return
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); setIndex(i => Math.max(0, Math.min(rows.length-1, i+(e.key === 'ArrowRight' ? 1 : -1)))) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [rows.length])

  async function action(kind: 'train' | 'test' | 'save') {
    setBusy(true); setError('')
    try {
      const result = await cnnAction(kind, kind === 'train' ? { variant, epochs, num_workers: workers, batch_size: batch, learning_rate: rate, seed: 42 } : { run_id: id })
      if (result.job_id) setJob(await getModelJob(result.job_id))
      if (kind === 'train') { setChoice(''); setIndex(0) }
      setRevision(r => r+1)
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }

  return <section className="cnn-page"><div className="page-intro model-writeup-intro"><h2>CNN · ResNet-18</h2><CNNWriteup /></div>
    <section className="panel"><h3>Architecture</h3><p>Whole-plate variants: 4 outputs. Patch variants: the same backbone with either 2 outputs (good / defect) or all 4 classes per patch.</p><div className="cnn-architecture">{architecture.map(([name, shape, description]) => <div key={name}><strong>{name}</strong><code>{shape}</code><small>{description}</small></div>)}</div><p className="feature-note">Each residual block adds a shortcut to two 3×3 convolutions. Input processing stops at the segmented plate, before glare removal, CLAHE and contrast. The 224×224 center crop may lose small scratches or edge detail.</p>
      <div className="model-controls"><label>Variant <select value={variant} onChange={e => setVariant(e.target.value as typeof variant)}><option value="standard">Standard ResNet</option><option value="defect_weighted">Defect-weighted ResNet (1 / 2 / 2 / 2)</option><option value="patch">Patch ResNet · good / defect</option><option value="patch_multiclass">Patch ResNet · four classes</option></select></label><label>Epochs <input type="number" min="1" max="200" value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label><label>Batch size <input type="number" min="2" max="128" value={batch} onChange={e => setBatch(Number(e.target.value))} /></label><label>Data workers <select value={workers} onChange={e => setWorkers(Number(e.target.value))}>{[0, 1, 2, 4, 8].map(n => <option key={n} value={n}>{n === 0 ? "0 (synchronous)" : n}</option>)}</select></label><label>Learning rate <input type="number" min="0.000001" max="0.1" step="0.0001" value={rate} onChange={e => setRate(Number(e.target.value))} /></label><button disabled={busy || !!job || epochs < 1 || batch < 2 || rate <= 0} onClick={() => void action('train')}>Train ResNet</button><button disabled={busy || !!job || !run?.model_available} onClick={() => void action('test')}>Test held-out images</button></div>
      {(variant === 'patch' || variant === 'patch_multiclass') && <p className="feature-note">Patch CNN training can take a while because each plate produces many overlapping patches. It takes longer than whole plate training, especially on CPU. Keep the server running and your computer awake until it finishes.</p>}
      <p className="feature-note">The defect-weighted variant gives good plates weight 1 and each defect class weight 2. All variants split whole images first. Patch ResNet uses 64px patches with 32px overlap, a binary or four-class head, and ground-truth masks for patch labels. In four-class mode, clean patches are good and annotated patches use the image’s defect type. Any annotated defect pixel inside the plate makes a positive training patch. Patches are resized to 224px without cropping. Segmented plates are cached once per job; parallel workers load and transform batches. Temporary caches are removed when the job ends. Training never uses the held-out set. First training downloads pretrained weights if needed. Metrics and visualizations are saved in CV/results/data/cnn; use Save model to retain the selected checkpoint.</p>
      {job && <p role="status">{job.phase} {job.total > 0 && `· ${job.done}/${job.total}`}</p>}{error && <p role="alert">{error}</p>}
    </section>
    <section className="panel"><div className="panel-heading"><h3>Performance summary</h3>{run && <label>Run <select value={id} onChange={e => { setChoice(e.target.value); setIndex(0) }}>{runs.map(r => <option key={r.run_id} value={r.run_id}>{variantName(r)} · {new Date(r.created_at).toLocaleString()} · {r.config.epochs} epochs</option>)}</select></label>}</div>
      {!run ? <p>Train a model to see results.</p> : <><p><strong>{variantName(run)} ResNet</strong> · Loss weights: {run.class_names.map((name, i) => `${name}: ${run.class_weights?.[i] ?? 1}`).join(', ')}. Summary cross-entropy is unweighted. {run.config.variant === 'patch' ? 'Binary patch runs use max patch defect score, flagged at 0.5.' : run.config.variant === 'patch_multiclass' ? 'Four-class patches use argmax. Up to two bad patches are tolerated only when good has the highest mean probability across all patches (ties do not qualify). Otherwise the most confident non-good patch determines the image class.' : ''}</p><button disabled={busy || run.model_saved || !run.model_available} onClick={() => void action('save')}>{run.model_saved ? 'Model saved' : run.model_available ? 'Save model' : 'Results only'}</button><div className="model-table-scroll"><table><thead><tr><th>Split</th><th>Images</th><th>Accuracy</th><th>Cross-entropy</th><th>Macro precision</th><th>Macro recall</th><th>Macro F1</th></tr></thead><tbody>{([['Training', run.training], ['Held-out test', run.test]] as const).map(([name, m]) => <tr key={name}><th>{name}</th>{m ? <><td>{score(m, 'macro avg')?.support}</td><td>{percent(m.accuracy)}</td><td>{m.loss.toFixed(4)}</td><td>{percent(score(m, 'macro avg')?.precision ?? 0)}</td><td>{percent(score(m, 'macro avg')?.recall ?? 0)}</td><td>{percent(score(m, 'macro avg')?.['f1-score'] ?? 0)}</td></> : <td colSpan={6}>Click Test held-out images.</td>}</tr>)}</tbody></table></div>
        {run.test?.patch_metrics && <p>Held-out patch metrics · Accuracy: {percent(run.test.patch_metrics.accuracy)} · Macro recall: {percent(score(run.test.patch_metrics, 'macro avg')?.recall ?? 0)} · Macro precision: {percent(score(run.test.patch_metrics, 'macro avg')?.precision ?? 0)} · Macro F1: {percent(score(run.test.patch_metrics, 'macro avg')?.['f1-score'] ?? 0)}</p>}
        {run.test && <p>Training minus test accuracy: {((run.training.accuracy-run.test.accuracy)*100).toFixed(1)} percentage points. A large positive gap can indicate overfitting; training history is not a validation curve.</p>}
        <div className="cnn-columns"><div><h4>Class performance and split counts</h4><div className="model-table-scroll"><table><thead><tr><th>Class</th><th>Train / test</th><th>Test correct / total</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{run.class_names.map((name, i) => { const m = run.test && score(run.test, name); return <tr key={name}><th>{name}</th><td>{run.split_counts.train[name]} / {run.split_counts.test[name]}</td><td>{m ? `${run.test!.confusion_matrix[i][i]}/${m.support} (${percent(m.recall)})` : '—'}</td><td>{m ? percent(m.precision) : '—'}</td><td>{m ? percent(m.recall) : '—'}</td><td>{m ? percent(m['f1-score']) : '—'}</td></tr> })}</tbody></table></div></div>
          {run.test && <div><h4>Test confusion matrix</h4><p>Rows: actual class. Columns: prediction.</p><div className="model-table-scroll"><table><thead><tr><th>Actual ↓ / Predicted →</th>{run.class_names.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{run.test.confusion_matrix.map((row, i) => <tr key={i}><th>{run.class_names[i]}</th>{row.map((n, j) => <td key={j} style={{ background: `rgba(35,123,148,${n/Math.max(1, ...row)*.3})` }}>{n}</td>)}</tr>)}</tbody></table></div></div>}</div></>}
    </section>
    <section className="panel"><h3>Learned weights</h3>{run && current ? <WeightView run={run} weights={current.weights} /> : <p>{run ? 'Loading weights…' : 'Train a model to inspect its filters and classifier weights.'}</p>}</section>
    <section className="panel"><h3>Individual predictions</h3>{!rows.length ? <p>Test the selected run to inspect held-out predictions.</p> : <><div className="model-controls"><button disabled={index <= 0} onClick={() => setIndex(i => i-1)}>← Previous</button><select aria-label="Test image" value={index} onChange={e => setIndex(Number(e.target.value))}>{rows.map((r, i) => <option key={r.path} value={i}>{r.correct ? 'Correct' : 'Wrong'} · {r.path.split('/').slice(-2).join('/')}</option>)}</select><button disabled={index >= rows.length-1} onClick={() => setIndex(i => i+1)}>Next →</button><span>{index+1}/{rows.length} · Left/right arrow keys switch images</span></div>
      {selected && <><p><strong>{selected.correct ? 'Correct' : 'Wrong'}</strong> · Actual: {selected.actual} · Predicted: {selected.prediction}</p><div className="cnn-predictions"><figure><figcaption>Original · {selected.path.split('/').at(-1)}</figcaption><img src={getImageUrl(selected.path.split('/')[0], selected.path.split('/')[1], selected.path.split('/').slice(2).join('/'))} alt="Original test plate" /></figure><figure><figcaption>{isPatch(run) ? 'Full segmented plate' : 'Segmented input after transforms'}</figcaption>{view ? <img src={view.input_image} alt="Exact segmented and cropped CNN input" /> : <p>Loading inspection…</p>}</figure><figure><figcaption>{isPatch(run) ? 'Localized patch predictions' : 'Predicted-class activation map'}</figcaption>{view && <img src={view.activation_image} alt="Class activation overlay on CNN input" />}</figure></div><p className="feature-note">{run?.config.variant === 'patch_multiclass' ? 'Boxes show predicted defect types: orange = major rust, blue = scratches, purple = total rust. Each patch uses its highest-scoring class. Clean patches are not outlined. These are patch classifications, not pixel masks.' : run?.config.variant === 'patch' ? 'Red boxes are patches predicted defective. A plate is flagged when any eligible patch scores at least 0.5. Overlapping patch scores are shown below; these are localized classifications, not pixel masks. Regions with less than 50% plate coverage are not scored.' : 'The heatmap shows positive evidence for the predicted class on a coarse 7×7 feature grid. It is not a defect segmentation mask. The overlay aligns with the cropped CNN input.'}</p>
        <div className="model-table-scroll"><table><thead><tr>{run?.class_names.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody><tr>{selected.probabilities.map((p, i) => <td key={i}>{percent(p)}</td>)}</tr></tbody></table></div><p className="feature-note">Scores are not calibrated probabilities. Binary patch runs show the maximum defect score. Four-class patch runs show mean scores for the good override, otherwise the selected patch’s scores.</p>
        {view?.patches && <details open><summary>Patch predictions ({view.patches.length})</summary><div className="model-table-scroll" style={{maxHeight: 320}}><table><thead><tr><th>Location (x, y)</th><th>{run?.config.variant === "patch_multiclass" ? "Predicted-class score" : "Defect score"}</th><th>Prediction</th><th>Ground truth</th>{run?.config.variant === "patch_multiclass" && run.class_names.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{view.patches.map((p, i) => <tr key={i}><td>{p.left}, {p.top}</td><td>{percent(p.score)}</td><td>{p.prediction ?? (p.anomalous ? "Defect" : "Good")}</td><td>{p.actual ?? (p.target ? "Defect" : "Good")}</td>{p.probabilities?.map((value, j) => <td key={j}>{percent(value)}</td>)}</tr>)}</tbody></table></div></details>}
        {view && !view.patches && <><h4>Eight most active final-layer channels</h4><div className="cnn-feature-maps">{view.feature_maps.map(f => <figure key={f.channel}><img src={f.image} alt={`Activation channel ${f.channel}`} /><figcaption>Channel {f.channel} · {f.activation.toFixed(3)}</figcaption></figure>)}</div><p>Each channel is scaled separately for display. High activation is not the same as positive class contribution.</p><details><summary>All 512 pooled feature values</summary><div className="cnn-feature-values">{view.features.map((v, i) => <span key={i}>{i}: {v.toFixed(4)}</span>)}</div></details></>}
      </>}</>}
    </section>
  </section>
}
