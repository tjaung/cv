import { useEffect, useId, useState } from 'react'
import { getImageSets, getImages, getImageUrl, getTestGroundTruth } from './api'
import type { ImageSets } from './api'
import './App.css'

function FolderIcon() {
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10H3V7Z" fill="#f7d680" stroke="#bc9135" strokeWidth="1.5" /></svg>
}

function ImagePreview({ src, name, thumbnail = false }: { src: string; name: string; thumbnail?: boolean }) {
  const [failed, setFailed] = useState(false)
  return failed
    ? <p className="empty" role="alert">Could not load this image.</p>
    : <img className={thumbnail ? 'thumbnail' : 'preview-image'} src={src} alt={name} loading={thumbnail ? 'lazy' : 'eager'} onError={() => setFailed(true)} />
}

function hasGroundTruth(split: string, name: string) {
  return split === 'test' && name.split('/').length === 2 && !name.startsWith('good/')
}

function OpacitySlider({ opacity, onChange, disabled = false, grid = false }: {
  opacity: number
  onChange: (opacity: number) => void
  disabled?: boolean
  grid?: boolean
}) {
  const sliderId = useId()
  return <div className="overlay-controls">
    <div className="overlay-label"><label htmlFor={sliderId}>{grid ? 'Ground truth opacity · all images' : 'Ground truth opacity'}</label><output htmlFor={sliderId}>{opacity}%</output></div>
    <input id={sliderId} type="range" min="0" max="100" step="1" value={opacity} disabled={disabled}
      onChange={(event) => onChange(Number(event.target.value))} aria-valuetext={`${opacity}% ground truth opacity`} />
    <div className="overlay-label overlay-hints"><span>Test image</span><span>Ground truth</span></div>
  </div>
}

function TestThumbnail({ imageSet, name, opacity }: { imageSet: string; name: string; opacity: number }) {
  const [failed, setFailed] = useState(false)
  // Match the server's mask naming convention; fetch only the mask, not a second test image.
  const maskName = name.replace(/\.[^/.]+$/, '_mask.png')
  return <>
    <div className="comparison-image">
      <ImagePreview src={getImageUrl(imageSet, 'test', name)} name={name} thumbnail />
      {!failed && <img className="thumbnail ground-truth-overlay" src={getImageUrl(imageSet, 'ground_truth', maskName)}
        alt={`Ground truth for ${name}`} loading="lazy" style={{ opacity: opacity / 100 }} onError={() => setFailed(true)} />}
    </div>
    {failed && <span className="mask-error">Ground truth unavailable</span>}
  </>
}

function TestImagePreview({ imageSet, name, opacity, onOpacityChange }: {
  imageSet: string
  name: string
  opacity: number
  onOpacityChange: (opacity: number) => void
}) {
  const [mask, setMask] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    const [defect, imageName] = name.split('/')
    getTestGroundTruth(imageSet, defect, imageName).then((pair) => {
      if (!active) return
      if (pair.ground_truth) setMask(pair.ground_truth.data_url)
      else setError('No ground truth is available for this image.')
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'Could not load ground truth.')
    })
    return () => { active = false }
  }, [imageSet, name])

  return <>
    <OpacitySlider opacity={opacity} onChange={onOpacityChange} disabled={!ready || !!error} />
    {(!ready || error) && <div className="overlay-status">
      {error ? <p className="overlay-message" role="alert">{error}</p>
        : !ready && <p className="overlay-message" role="status">Loading ground truth…</p>}
    </div>}
    <div className="comparison-image">
      <ImagePreview src={getImageUrl(imageSet, 'test', name)} name={name} />
      {mask && !error && <img className="preview-image ground-truth-overlay" src={mask} alt={`Ground truth for ${name}`}
        style={{ opacity: opacity / 100 }} onLoad={() => setReady(true)}
        onError={() => setError('Could not display the ground-truth image.')} />}
    </div>
  </>
}

function Directory({ path, sets, navigate }: {
  path: string[]
  sets: ImageSets
  navigate: (path: string[]) => void
}) {
  const [images, setImages] = useState<string[] | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<'preview' | 'grid'>('grid')
  const [gridOpacity, setGridOpacity] = useState(0)
  const [previewOpacity, setPreviewOpacity] = useState(0)
  const [attempt, setAttempt] = useState(0)
  const [imageSet, split] = path

  useEffect(() => {
    if (!imageSet || !split) return
    let active = true
    getImages(imageSet, split).then((names) => {
      if (active) setImages(names)
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'Could not load images.')
    })
    return () => { active = false }
  }, [imageSet, split, attempt])

  const prefix = path.length > 2 ? `${path.slice(2).join('/')}/` : ''
  const folders = new Set<string>()
  const files: string[] = []
  if (path.length === 0) Object.keys(sets).forEach((name) => folders.add(name))
  else if (path.length === 1) sets[imageSet]?.forEach((name) => folders.add(name))
  else images?.forEach((name) => {
    if (!name.startsWith(prefix)) return
    const remaining = name.slice(prefix.length)
    if (remaining.includes('/')) folders.add(remaining.split('/')[0])
    else files.push(name)
  })
  const loading = path.length >= 2 && images === null && !error
  const canOverlayGrid = files.some((name) => hasGroundTruth(split, name))
  files.sort()
  const selectedIndex = selected === null ? -1 : files.indexOf(selected)
  const previousImage = selectedIndex > 0 ? files[selectedIndex - 1] : undefined
  const nextImage = selectedIndex >= 0 ? files[selectedIndex + 1] : undefined

  useEffect(() => {
    if (view !== 'preview' || selected === null) return

    function onKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return
      const target = event.target
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest('input, textarea, select, [role="slider"]'))) return
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return

      event.preventDefault()
      const image = event.key === 'ArrowLeft' ? previousImage : nextImage
      if (image !== undefined) setSelected(image)
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [view, selected, previousImage, nextImage])

  function selectImage(name: string) {
    setSelected(name)
    setView('preview')
  }

  return <div className="workspace">
    <section className="panel file-browser" aria-label="Folder contents">
      <div className="panel-heading"><h2>Files</h2><span>{loading ? 'Loading…' : `${folders.size} folders · ${files.length} images`}</span></div>
      {error ? <div className="empty" role="alert"><p>{error}</p><button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Try again</button></div>
        : loading ? <p className="empty" role="status">Loading images…</p>
        : folders.size === 0 && files.length === 0 ? <p className="empty">This folder is empty.</p>
        : <ul className="entries">
          {[...folders].sort().map((name) => <li key={`folder:${name}`}><button className="entry" onClick={() => navigate([...path, name])}><FolderIcon /><span>{name}</span><span className="entry-kind">Folder</span><span aria-hidden="true">›</span></button></li>)}
          {files.map((name) => <li key={name}><button className={`entry ${selected === name ? 'selected' : ''}`} aria-pressed={selected === name} onClick={() => selectImage(name)}><span className="file-icon" aria-hidden="true">▧</span><span>{name.slice(prefix.length)}</span><span className="entry-kind">Image</span></button></li>)}
        </ul>}
    </section>
    <section className="panel preview" aria-label="Image preview">
      <div className="panel-heading preview-heading">
        <div><h2>{view === 'grid' ? 'Folder images' : 'Image preview'}</h2><p className="preview-caption">{view === 'grid' ? `${files.length} images` : selected || 'No image selected'}</p></div>
        <div className="view-switch" aria-label="Image view">
          {view === 'preview' && <>
            <button disabled={previousImage === undefined} onClick={() => previousImage !== undefined && setSelected(previousImage)} title="Previous image (Left arrow)">← Previous</button>
            <button disabled={nextImage === undefined} onClick={() => nextImage !== undefined && setSelected(nextImage)} title="Next image (Right arrow)">Next →</button>
            <button onClick={() => setView('grid')}>← Back to grid</button>
          </>}
        </div>
      </div>
      {view === 'grid' ? <>
        <OpacitySlider opacity={gridOpacity} onChange={setGridOpacity} disabled={!canOverlayGrid} grid />
        {!canOverlayGrid && <p className="overlay-status overlay-message">Open a defective test image folder to compare ground truth.</p>}
        {!files.length && <div className="empty preview-empty"><p>{loading ? 'Loading images…' : error ? 'Images could not be loaded. Try again from the file browser.' : folders.size ? 'Open a folder to see its images here.' : 'No images in this folder.'}</p></div>}
        <ul className="image-grid">
        {files.map((name) => <li key={name}><button className="image-card" onClick={() => selectImage(name)} aria-label={`Preview ${name.slice(prefix.length)}`}>
          {hasGroundTruth(split, name)
            ? <TestThumbnail imageSet={imageSet} name={name} opacity={gridOpacity} />
            : <ImagePreview src={getImageUrl(imageSet, split, name)} name={name} thumbnail />}
          <span>{name.slice(prefix.length)}</span>
        </button></li>)}
      </ul></> : selected ? (hasGroundTruth(split, selected)
        ? <TestImagePreview key={selected} imageSet={imageSet} name={selected} opacity={previewOpacity} onOpacityChange={setPreviewOpacity} />
        : <ImagePreview key={selected} src={getImageUrl(imageSet, split, selected)} name={selected} />)
        : <div className="empty preview-empty"><span aria-hidden="true">▧</span><p>Select an image to preview it, or show all images in this folder.</p></div>}
    </section>
  </div>
}

function App() {
  const [sets, setSets] = useState<ImageSets | null>(null)
  const [error, setError] = useState('')
  const [path, setPath] = useState<string[]>([])
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let active = true
    getImageSets().then((data) => {
      if (active) setSets(data)
    }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'Could not load image sets.')
    })
    return () => { active = false }
  }, [attempt])

  return <main>
    <header><p className="eyebrow">DATASET EXPLORER</p><h1>Anomaly dataset</h1><p>Browse folders and select an image to take a closer look.</p></header>
    <nav className="breadcrumbs" aria-label="Folder navigation">
      <button className="back" disabled={!path.length} onClick={() => setPath(path.slice(0, -1))} aria-label="Go to parent folder">↑</button>
      <button onClick={() => setPath([])} aria-current={!path.length ? 'location' : undefined}>anomaly_dataset</button>
      {path.map((part, index) => <span className="crumb" key={index}><span aria-hidden="true">/</span><button aria-current={index === path.length - 1 ? 'location' : undefined} onClick={() => setPath(path.slice(0, index + 1))}>{part}</button></span>)}
    </nav>
    {error ? <div className="panel empty" role="alert"><p>{error}</p><p>Check that the server is running on port 8000.</p><button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Try again</button></div>
      : sets ? <Directory key={JSON.stringify(path)} path={path} sets={sets} navigate={setPath} />
      : <p className="panel empty" role="status">Loading image sets…</p>}
  </main>
}

export default App
