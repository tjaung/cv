import ClassifierPatchImage from './ClassifierPatchImage'
import { useEffect, useRef, useState } from 'react'
import { classifierRequest, getImageUrl } from './api'
import type { Model, Summary } from './Classifiers'

type Prediction = { model_id: string; prediction: string; correct: boolean }
type ReviewResult = { run_id: string; image_index: number; path: string; label: string; membership: 'training' | 'holdout'; predictions: Prediction[] }
type Session = { used: Record<string, boolean>; results: Record<string, ReviewResult>; selected: number | null }
const modelName = (m: Model) => Object.entries(m.config).map(([key, value]) => key === 'kind' ? String(value).toUpperCase() : `${key}=${value}`).join(' · ')
const fraction = (correct: number, total: number) => total ? `${correct}/${total} (${(correct / total * 100).toFixed(1)}%)` : '—'

export default function RandomClassifierReview({ summary, training }: { summary: Summary; training: boolean }) {
  const inventory = [...summary.train, ...summary.test]
  const modelIds = summary.models.map(m => m.id).sort().join(',')
  const storageKey = `classifier-review-v1:${summary.run_id}:${modelIds}`
  const blank = (): Session => ({ used: Object.fromEntries(inventory.map(s => [s.path, false])), results: {}, selected: null })
  const [session, setSession] = useState<Session>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? 'null') as Session | null
      if (saved && Object.keys(saved.used).length === inventory.length && inventory.every(s => typeof saved.used[s.path] === 'boolean' && (!saved.used[s.path] || saved.results[s.path]?.predictions.length === summary.models.length)) && (saved.selected === null || (Number.isInteger(saved.selected) && saved.selected >= 0 && saved.selected < inventory.length))) return saved
    } catch { /* A missing or invalid saved session starts fresh. */ }
    return blank()
  })
  const [explanationModel, setExplanationModel] = useState('')
  const [error, setError] = useState('')
  const [storageError, setStorageError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const section = useRef<HTMLElement>(null)
  const advancing = useRef(false)
  const followed = summary.models.find(m => m.id === explanationModel)
  const displayedModel = followed?.id ?? summary.models[0].id
  const selected = session.selected
  const sample = selected === null ? null : inventory[selected]
  const current = sample ? session.results[sample.path] : undefined
  const currentPath = sample?.path
  const needsScoring = !!sample && !current
  const completed = Object.values(session.used).filter(Boolean).length
  const finished = completed === inventory.length
  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(session)) }
    catch {
      const timer = window.setTimeout(() => setStorageError('Browser storage is full or unavailable. This review will last only while the page stays open; download results to keep them.'), 0)
      return () => window.clearTimeout(timer)
    }
  }, [session, storageKey])
  useEffect(() => {
    if (selected === null || !needsScoring || training) return
    const controller = new AbortController()
    classifierRequest<ReviewResult>(`review/?run_id=${summary.run_id}&image_index=${selected}`, undefined, controller.signal).then(result => {
      if (controller.signal.aborted) return
      if (result.path !== currentPath || result.run_id !== summary.run_id || result.predictions.length !== modelIds.split(',').length || !modelIds.split(',').every(id => result.predictions.some(p => p.model_id === id))) throw new Error('Model inventory changed. Wait for training to finish before reviewing.')
      setSession(s => ({ ...s, used: { ...s.used, [result.path]: true }, results: { ...s.results, [result.path]: result } }))
    }).catch((reason: unknown) => { if (!controller.signal.aborted) setError(String(reason)) })
    return () => controller.abort()
  }, [selected, needsScoring, currentPath, summary.run_id, modelIds, training, attempt])
  const next = () => {
    if (advancing.current || training || needsScoring || finished) return
    advancing.current = true
    setError('')
    setSession(s => {
      const remaining = inventory.map((item, index) => ({ item, index })).filter(({ item }) => !s.used[item.path])
      if (!remaining.length) return s
      return { ...s, selected: remaining[Math.floor(Math.random() * remaining.length)].index }
    })
    section.current?.focus({ preventScroll: true })
    // Prevent double clicks before React commits the newly selected image.
    window.setTimeout(() => { advancing.current = false }, 0)
  }
  const download = () => {
    const blob = new Blob([JSON.stringify({ run_id: summary.run_id, models: summary.models.map(m => ({ id: m.id, config: m.config })), ...session }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `classifier-review-${summary.run_id}.json`; anchor.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const rankings = summary.models.map(model => {
    const results = Object.values(session.results)
    const count = (items: ReviewResult[]) => items.filter(r => r.predictions.find(p => p.model_id === model.id)?.correct).length
    return { model, correct: count(results), train: count(results.filter(r => r.membership === 'training')), test: count(results.filter(r => r.membership === 'holdout')), perClass: Object.fromEntries(summary.classes.map(c => [c, count(results.filter(r => r.label === c))])) }
  }).sort((a,b) => b.correct-a.correct || modelName(a.model).localeCompare(modelName(b.model)))
  return <section ref={section} tabIndex={0} className="panel random-classifier-review" aria-label="Random image review" onKeyDown={event => {
    if (!['ArrowLeft','ArrowRight'].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || (event.target instanceof Element && event.target.closest('input,select,textarea,[contenteditable]'))) return
    event.preventDefault(); event.stopPropagation()
    if (!event.repeat) next()
  }}>
    <div className="panel-heading"><h3>Test random image</h3><strong role="status">{completed} / {inventory.length} used</strong></div>
    <p className="feature-note">Random order without repeats across this run’s original train and test images. All {summary.models.length} classifiers participate, regardless of the filter above. Focus this section and press ← or → for the next unused image. Progress is saved in this browser.</p>
    <p className="feature-note">Training images were seen during fitting. The final ranking shows training and holdout accuracy separately; the combined result is a dataset review, not unseen-data performance.</p>
    {training && <p className="feature-note" role="status">Wait for classifier training to finish before continuing the review.</p>}
    {storageError && <p role="alert" className="feature-note">{storageError}</p>}
    <div className="model-controls model-training"><button disabled={training || needsScoring || finished} onClick={next}>{finished ? 'All images reviewed' : selected === null ? 'Start random review' : 'Next random image →'}</button><button disabled={!completed} onClick={download}>Download review results</button>{finished && <button onClick={() => { setSession(blank()); setError(''); section.current?.focus({ preventScroll: true }) }}>Start a new review</button>}</div>
    <div className="model-table-scroll"><table><caption>Images used by actual class</caption><thead><tr><th>Class</th><th>Used / total</th><th>Remaining</th></tr></thead><tbody>{summary.classes.map(label => {
      const items = inventory.filter(s => s.label === label), used = items.filter(s => session.used[s.path]).length
      return <tr key={label}><th>{label}</th><td>{used} / {items.length}</td><td>{items.length-used}</td></tr>
    })}</tbody></table></div>
    {followed && <p className="feature-note">Following {modelName(followed)} · pinned first with automatic patch explanations. <button onClick={() => setExplanationModel('')}>Stop following</button></p>}
    {!followed && <p className="feature-note">Click a model name below to follow it across random images and automatically explain its predicted defect patches.</p>}
    {sample && <div className="random-classifier-inspection"><figure><ClassifierPatchImage key={`${displayedModel}:${selected}`} autoExplain={!!followed} run={summary.run_id} model={displayedModel} index={selected!} src={getImageUrl('metal_plate', sample.original_split, sample.path.split('/').slice(2).join('/'))} alt={sample.path} /><figcaption><strong>{sample.path.split('/').pop()}</strong><p>Actual: {sample.label} · {selected! < summary.train.length ? 'Training image (seen during fitting)' : 'Holdout image'}</p><span>{sample.path}</span></figcaption></figure><div>
      {needsScoring && !error && <p role="status">Classifying with every model…</p>}
      {error && <p role="alert">{error} <button onClick={() => { setError(''); setAttempt(a => a+1) }}>Retry this image</button></p>}
      {current && <div className="model-table-scroll"><table><thead><tr><th>Model</th><th>Prediction</th><th>Result</th></tr></thead><tbody>{[...current.predictions].sort((a,b) => Number(b.model_id === followed?.id)-Number(a.model_id === followed?.id) || Number(b.correct)-Number(a.correct)).map(p => <tr key={p.model_id} aria-selected={followed?.id === p.model_id}><th><button aria-pressed={explanationModel === p.model_id} onClick={() => setExplanationModel(p.model_id)}>{followed?.id === p.model_id ? 'Following · ' : ''}{modelName(summary.models.find(m => m.id === p.model_id)!)}</button></th><td>{p.prediction}</td><td className={p.correct ? 'classification-right' : 'classification-wrong'}>{p.correct ? 'Right' : 'Wrong'}</td></tr>)}</tbody></table></div>}
    </div></div>}
    <details className="model-training"><summary>Image usage dictionary ({completed} true / {inventory.length-completed} false)</summary><div className="model-table-scroll random-usage"><table><thead><tr><th>Image path</th><th>Used</th></tr></thead><tbody>{inventory.map(s => <tr key={s.path}><td>{s.path}</td><td>{String(session.used[s.path])}</td></tr>)}</tbody></table></div></details>
    {finished && <section className="model-training" aria-label="Final classifier ranking"><h3>Final ranking · entire dataset review</h3><p>Ranked by total correctly classified images. Models with equal totals share a rank.</p><div className="model-table-scroll"><table><thead><tr><th>Rank</th><th>Model</th><th>All images correct</th><th>Training correct</th><th>Holdout correct</th>{summary.classes.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{rankings.map(r => <tr key={r.model.id}><td>{1+rankings.filter(other => other.correct>r.correct).length}</td><th>{modelName(r.model)}</th><td>{fraction(r.correct,inventory.length)}</td><td>{fraction(r.train,summary.train.length)}</td><td>{fraction(r.test,summary.test.length)}</td>{summary.classes.map(c => <td key={c}>{fraction(r.perClass[c],inventory.filter(s=>s.label===c).length)}</td>)}</tr>)}</tbody></table></div></section>}
  </section>
}
