import { useEffect, useRef, useState } from 'react'
import PreprocessingWriteup from './PreprocessingWriteup'
import { getImages, getPreprocessedImage, getPreprocessingSteps } from '../../api'
import type { PreprocessingStep, PreprocessingPipeline } from '../../api'

type Split = 'train' | 'test'
interface PlateImage { path: string; name: string; split: Split }
interface Dataset { images: PlateImage[]; steps: Record<PreprocessingPipeline, PreprocessingStep[]> }

function StageImage({ image, step, pipeline, allowBorderTouching, thresholdOffset, glareCutoff, surroundingRadius, blendWidth, thumbnail = false }: {
  image: PlateImage
  step: number
  pipeline: PreprocessingPipeline
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
      getPreprocessedImage(image.path, step, allowBorderTouching, controller.signal, thresholdOffset, glareCutoff, surroundingRadius, blendWidth, 1.5, pipeline).then((data) => {
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
  }, [image.path, step, pipeline, allowBorderTouching, thumbnail, attempt, thresholdOffset, glareCutoff, surroundingRadius, blendWidth])

  return <div ref={container} className={`stage-image ${thumbnail ? 'stage-thumbnail' : 'stage-single'}`}>
    {error ? <div className="stage-error"><p>{error}</p>{!thumbnail && <button onClick={() => { setResult(null); setError(''); setAttempt(attempt + 1) }}>Try again</button>}</div>
      : result ? <>
        <img src={result.url} alt={`${image.name}, preprocessing step ${step}`} onError={() => setError('Could not display the processed image.')} />
        {pipeline !== 'raw' && step >= (pipeline === 'normalized' ? 5 : 4) && result.threshold !== null && <span className="threshold-readout">ISODATA {result.threshold.toFixed(1)}</span>}
        {pipeline !== 'raw' && step >= (pipeline === 'normalized' ? 7 : 6) && !result.plateFound && <span className="no-plate">No plate found{!thumbnail && '. Try allowing border-connected regions above, or inspect the ISODATA and Cleanup views.'}</span>}
      </>
      : <span className="stage-loading">{step === 0 ? 'Loading image…' : 'Processing…'}</span>}
  </div>
}

export default function Preprocessing() {
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [split, setSplit] = useState<'all' | Split>('all')
  const [pipeline, setPipeline] = useState<PreprocessingPipeline>('segmentation')
  const [stepChoices, setStepChoices] = useState({ segmentation: 0, full: 0, normalized: 0, raw: 0 })
  const step = stepChoices[pipeline]
  const setStep = (number: number) => setStepChoices(previous => ({ ...previous, [pipeline]: number }))
  const [selected, setSelected] = useState<string | null>(null)
  const [allowBorderTouching, setAllowBorderTouching] = useState(true)
  const appliedOffset = 20
  const glareCutoff = 220
  const surroundingRadius = 5
  const blendWidth = 3

  useEffect(() => {
    let active = true
    Promise.all([getImages('metal_plate', 'train'), getImages('metal_plate', 'test'), getPreprocessingSteps('segmentation'), getPreprocessingSteps('full'), getPreprocessingSteps('normalized'), getPreprocessingSteps('raw')])
      .then(([train, test, segmentation, full, normalized, raw]) => {
        if (!active) return
        const images = (['train', 'test'] as const).flatMap((part) =>
          (part === 'train' ? train : test).map((name) => ({ path: `metal_plate/${part}/${name}`, name, split: part })))
        setDataset({ images, steps: { segmentation, full, normalized, raw } })
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
  const activeStep = dataset?.steps[pipeline].find((item) => item.number === step)

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
    <div className="page-intro preprocessing-intro"><p className="eyebrow">METAL PLATE · TRAINING & TEST</p><h2>Preprocessing</h2>
      <PreprocessingWriteup />
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
            <label className="cropped-option"><input type="checkbox" disabled={pipeline === 'raw'} checked={allowBorderTouching} onChange={(event) => setAllowBorderTouching(event.target.checked)} /> Allow border-connected regions</label>
          </div>
          <div className="view-switch" role="tablist" aria-label="Preprocessing pipeline">
            {([['segmentation', 'Segmentation · CNN'], ['full', 'Full · no normalization'], ['normalized', 'Full · normalization'], ['raw', 'Raw · features only']] as const).map(([value, label]) => <button key={value} id={`pipeline-tab-${value}`} role="tab" aria-selected={pipeline === value} aria-controls="pipeline-preview" onClick={() => setPipeline(value)}>{label}</button>)}
          </div>
          <p className="step-description">{pipeline === 'raw' ? 'No image preprocessing or segmentation. Anomaly features use the entire image, including background.' : pipeline === 'normalized' ? 'Anomaly comparison: apply 10% luminance correction to the color branch, then run the full pipeline. Mask extraction still uses original grayscale.' : pipeline === 'segmentation'
            ? 'Build a mask from the grayscale branch, then apply it to the original color image. Whole-image CNNs use the segmented plate; patch CNNs extract overlapping 64×64 regions with stride 32. Patch boxes are input regions, not predictions.'
            : 'Continue after segmentation with glare detection, surrounding-color fill, boundary blending, plate blur, CLAHE and final contrast. The final color plate feeds LAB, Sobel, HOG and Frangi features for classical anomaly detectors and classifiers.'}</p>
          <div className="pipeline-steps" aria-label="Preprocessing steps">
            {dataset.steps[pipeline].map((item) => <button key={item.number} aria-pressed={step === item.number} title={item.description} onClick={() => setStep(item.number)}>
              <span className="step-number">{item.number}</span><span>{item.name}</span>
            </button>)}
          </div>
          <p className="step-description" aria-live="polite"><strong>{activeStep?.name}.</strong> {activeStep?.description}</p>
        </div>
        <div id="pipeline-preview" role="tabpanel" aria-labelledby={`pipeline-tab-${pipeline}`}>
        {current ? <>
          <div className="panel-heading preview-heading"><div><h3>{current.name.split('/').at(-1)}</h3><p className="preview-caption">{current.split} / {current.name} · {selectedIndex + 1} of {images.length}</p></div>
            <div className="view-switch">
              <button disabled={!previous} onClick={() => previous && setSelected(previous)} title="Previous image (Left arrow)">← Previous</button>
              <button disabled={!next} onClick={() => next && setSelected(next)} title="Next image (Right arrow)">Next →</button>
              <button onClick={() => setSelected(null)}>← Back to grid</button>
            </div>
          </div>
          <StageImage key={`${current.path}:${pipeline}:${step}:${allowBorderTouching}:${appliedOffset}:${glareCutoff}:${surroundingRadius}:${blendWidth}`} image={current} step={step} pipeline={pipeline} allowBorderTouching={allowBorderTouching} thresholdOffset={appliedOffset} glareCutoff={glareCutoff} surroundingRadius={surroundingRadius} blendWidth={blendWidth} />
        </> : images.length === 0 ? <p className="empty">No images in this split.</p>
          : <div className="preprocessing-grid-scroll">
            {(['train', 'test'] as const).filter((part) => split === 'all' || part === split).map((part) => <section key={part} aria-label={`${part === 'train' ? 'Training' : 'Test'} images`}>
              <div className="group-heading"><h3>{part === 'train' ? 'Training images' : 'Test images'}</h3><span>{images.filter((image) => image.split === part).length} images</span></div>
              <ul className="image-grid preprocessing-grid">
                {images.filter((image) => image.split === part).map((image) => <li key={image.path}>
                  <button className="image-card" aria-label={`View ${image.split}/${image.name}`} onClick={() => setSelected(image.path)}>
                    <StageImage key={`${image.path}:${pipeline}:${step}:${allowBorderTouching}:${appliedOffset}:${glareCutoff}:${surroundingRadius}:${blendWidth}`} image={image} step={step} pipeline={pipeline} allowBorderTouching={allowBorderTouching} thresholdOffset={appliedOffset} glareCutoff={glareCutoff} surroundingRadius={surroundingRadius} blendWidth={blendWidth} thumbnail />
                    <span className="processed-filename">{image.name.split('/').at(-1)}<small>{image.name.split('/').slice(0, -1).join('/')}</small></span>
                  </button>
                </li>)}
              </ul>
            </section>)}
          </div>}
        </div>
      </div>}
  </section>
}
