import { useEffect, useRef, useState } from 'react'
import { getImages, getPreprocessedImage, getPreprocessingSteps } from './api'
import type { PreprocessingStep } from './api'

type Split = 'train' | 'test'
interface PlateImage { path: string; name: string; split: Split }
interface Dataset { images: PlateImage[]; steps: PreprocessingStep[] }

function StageImage({ image, step, allowBorderTouching, thresholdOffset, glareCutoff, surroundingRadius, blendWidth, thumbnail = false }: {
  image: PlateImage
  step: number
  allowBorderTouching: boolean
  glareCutoff: number
  surroundingRadius: number
  blendWidth: number
  thresholdOffset: number
  thumbnail?: boolean
}) {
  const container = useRef<HTMLDivElement>(null)
  const [result, setResult] = useState<{ url: string; plateFound: boolean; threshold: number | null } | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let active = true
    let objectUrl: string | undefined
    let started = false
    const controller = new AbortController()
    const load = () => {
      if (started) return
      started = true
      getPreprocessedImage(image.path, step, allowBorderTouching, controller.signal, thresholdOffset, glareCutoff, surroundingRadius, blendWidth).then((data) => {
        if (!active) return
        objectUrl = URL.createObjectURL(data.blob)
        setResult({ url: objectUrl, plateFound: data.plateFound, threshold: data.threshold })
      }).catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : 'Could not process image.')
      })
    }
    // Process thumbnails only as they approach the visible part of the grid.
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        load()
        observer.disconnect()
      }
    }, { rootMargin: '250px' })
    if (thumbnail && container.current) observer.observe(container.current)
    else load()
    return () => {
      active = false
      observer.disconnect()
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [image.path, step, allowBorderTouching, thumbnail, attempt, thresholdOffset, glareCutoff, surroundingRadius, blendWidth])

  return <div ref={container} className={`stage-image ${thumbnail ? 'stage-thumbnail' : 'stage-single'}`}>
    {error ? <div className="stage-error"><p>{error}</p>{!thumbnail && <button onClick={() => { setResult(null); setError(''); setAttempt(attempt + 1) }}>Try again</button>}</div>
      : result ? <>
        <img src={result.url} alt={`${image.name}, preprocessing step ${step}`} onError={() => setError('Could not display the processed image.')} />
        {step >= 5 && result.threshold !== null && <span className="threshold-readout">ISODATA {result.threshold.toFixed(1)}</span>}
        {step >= 7 && !result.plateFound && <span className="no-plate">No plate found{!thumbnail && '. Try allowing border-connected regions above, or inspect the ISODATA and Cleanup views.'}</span>}
      </>
      : <span className="stage-loading">{step === 0 ? 'Loading image…' : 'Processing…'}</span>}
  </div>
}

export default function Preprocessing() {
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [split, setSplit] = useState<'all' | Split>('all')
  const [step, setStep] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [allowBorderTouching, setAllowBorderTouching] = useState(true)
  const appliedOffset = 20
  const glareCutoff = 220
  const surroundingRadius = 5
  const blendWidth = 3

  useEffect(() => {
    let active = true
    Promise.all([getImages('metal_plate', 'train'), getImages('metal_plate', 'test'), getPreprocessingSteps()])
      .then(([train, test, steps]) => {
        if (!active) return
        const images = (['train', 'test'] as const).flatMap((part) =>
          (part === 'train' ? train : test).map((name) => ({ path: `metal_plate/${part}/${name}`, name, split: part })))
        setDataset({ images, steps })
      }).catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : 'Could not load metal plate images.')
      })
    return () => { active = false }
  }, [attempt])

  const images = dataset?.images.filter((image) => split === 'all' || image.split === split) ?? []
  const selectedIndex = images.findIndex((image) => image.path === selected)
  const current = images[selectedIndex]
  const previous = selectedIndex > 0 ? images[selectedIndex - 1].path : undefined
  const next = selectedIndex >= 0 ? images[selectedIndex + 1]?.path : undefined
  const activeStep = dataset?.steps.find((item) => item.number === step)

  useEffect(() => {
    if (!selected) return
    const navigate = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      if (event.target instanceof HTMLElement && (event.target.isContentEditable || event.target.closest('input, select, textarea'))) return
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      event.preventDefault()
      const path = event.key === 'ArrowLeft' ? previous : next
      if (path) setSelected(path)
    }
    window.addEventListener('keydown', navigate)
    return () => window.removeEventListener('keydown', navigate)
  }, [selected, previous, next])

  return <section>
    <div className="page-intro"><p className="eyebrow">METAL PLATE · TRAINING & TEST</p><h2>Preprocessing</h2>
      <p>Normalize luminance, convert to grayscale, blur a copy, and darken edges using Sobel strength. Separate foreground with ISODATA, then fill enclosed holes using flood fill and apply one dilation followed by one erosion to close small gaps. Extract the plate region, trim its border with three 3×3 erosions, and apply it to the normalized, unblurred color image to preserve surface detail. Glare removal starts from the segmented color plate: grayscale → bright-pixel threshold → surrounding median-color fill → inward boundary blending → grayscale Gaussian blur → CLAHE → final Canny edges.</p>
      <p>Choose a numbered step to see its result across the grid. The largest foreground region is selected even when connected to the image border; turn off border connections for strict exclusion. Click an image for a closer look; your chosen step stays selected as you browse.</p>
    </div>
    {error ? <div className="panel empty" role="alert"><p>{error}</p><button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Try again</button></div>
      : !dataset ? <p className="panel empty" role="status">Loading metal plate images…</p>
      : <div className="panel preprocessing-panel">
        <div className="preprocessing-toolbar">
          <div className="preprocessing-toolbar-top">
            <div className="view-switch" aria-label="Image split">
              {(['all', 'train', 'test'] as const).map((value) => <button key={value} aria-pressed={split === value} onClick={() => { setSplit(value); setSelected(null) }}>
                {value === 'all' ? 'All images' : value === 'train' ? 'Training' : 'Test'} <span className="count">{dataset.images.filter((image) => value === 'all' || image.split === value).length}</span>
              </button>)}
            </div>
            <label className="cropped-option"><input type="checkbox" checked={allowBorderTouching} onChange={(event) => setAllowBorderTouching(event.target.checked)} /> Allow border-connected regions</label>
          </div>
          <div className="pipeline-steps" aria-label="Preprocessing steps">
            {dataset.steps.map((item) => <button key={item.number} aria-pressed={step === item.number} title={item.description} onClick={() => setStep(item.number)}>
              <span className="step-number">{item.number}</span><span>{item.name}</span>
            </button>)}
          </div>
          <p className="step-description" aria-live="polite"><strong>{activeStep?.name}.</strong> {activeStep?.description}</p>
        </div>
        {current ? <>
          <div className="panel-heading preview-heading"><div><h3>{current.name.split('/').at(-1)}</h3><p className="preview-caption">{current.split} / {current.name} · {selectedIndex + 1} of {images.length}</p></div>
            <div className="view-switch">
              <button disabled={!previous} onClick={() => previous && setSelected(previous)} title="Previous image (Left arrow)">← Previous</button>
              <button disabled={!next} onClick={() => next && setSelected(next)} title="Next image (Right arrow)">Next →</button>
              <button onClick={() => setSelected(null)}>← Back to grid</button>
            </div>
          </div>
          <StageImage key={`${current.path}:${step}:${allowBorderTouching}:${appliedOffset}:${glareCutoff}:${surroundingRadius}:${blendWidth}`} image={current} step={step} allowBorderTouching={allowBorderTouching} thresholdOffset={appliedOffset} glareCutoff={glareCutoff} surroundingRadius={surroundingRadius} blendWidth={blendWidth} />
        </> : images.length === 0 ? <p className="empty">No images in this split.</p>
          : <div className="preprocessing-grid-scroll">
            {(['train', 'test'] as const).filter((part) => split === 'all' || part === split).map((part) => <section key={part} aria-label={`${part === 'train' ? 'Training' : 'Test'} images`}>
              <div className="group-heading"><h3>{part === 'train' ? 'Training images' : 'Test images'}</h3><span>{images.filter((image) => image.split === part).length} images</span></div>
              <ul className="image-grid preprocessing-grid">
                {images.filter((image) => image.split === part).map((image) => <li key={image.path}>
                  <button className="image-card" aria-label={`View ${image.split}/${image.name}`} onClick={() => setSelected(image.path)}>
                    <StageImage key={`${image.path}:${step}:${allowBorderTouching}:${appliedOffset}:${glareCutoff}:${surroundingRadius}:${blendWidth}`} image={image} step={step} allowBorderTouching={allowBorderTouching} thresholdOffset={appliedOffset} glareCutoff={glareCutoff} surroundingRadius={surroundingRadius} blendWidth={blendWidth} thumbnail />
                    <span className="processed-filename">{image.name.split('/').at(-1)}<small>{image.name.split('/').slice(0, -1).join('/')}</small></span>
                  </button>
                </li>)}
              </ul>
            </section>)}
          </div>}
      </div>}
  </section>
}
