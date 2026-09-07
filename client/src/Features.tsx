import HOGView from './HOGView'
import FeaturesWriteup from './FeaturesWriteup'
import { useEffect, useState } from 'react'
import LBPView from './LBPView'
import { getFeatures, getFeatureHistograms, getImages } from './api'
import type { FeatureHistograms, FeatureMap, FeatureSource, FeatureSplit, ImageFeatures, HistogramChannel } from './api'

const colors = ['#235d44', '#bd4939', '#486bd4', '#a555a6', '#bd7a13', '#237f88']
const channelNames: Record<HistogramChannel, string> = { frangi_dark: 'Frangi · dark ridges', frangi_bright: 'Frangi · bright ridges', L: 'L* · lightness', a: 'a* · green → red', b: 'b* · blue → yellow', H: 'H · hue', S: 'S · saturation', V: 'V · brightness', magnitude: 'Sobel magnitude · edge strength', orientation: 'Sobel orientation · weighted by strength', hog: 'HOG · locally normalized orientations', lbp: 'Uniform LBP · texture patterns' }
const channelUnits: Record<HistogramChannel, string> = { frangi_dark: 'Ridge response', frangi_bright: 'Ridge response', L: 'L* value', a: 'a* value', b: 'b* value', H: 'Hue (degrees)', S: 'Saturation (%)', V: 'Value (%)', magnitude: 'Intensity change / pixel', orientation: 'Gradient orientation (degrees)', hog: 'Unsigned gradient orientation (degrees)', lbp: 'LBP code (9 = non-uniform)' }

function Heatmap({ map, transformed }: { map: FeatureMap; transformed: boolean }) {
  const [cell, setCell] = useState<{ row: number; col: number } | null>(null)
  return <article className="feature-card">
    <h3>{map.name}</h3>
    <div className="feature-map" onMouseLeave={() => setCell(null)} onMouseMove={(event) => {
      const rect = event.currentTarget.getBoundingClientRect()
      setCell({ row: Math.min(map.y.length - 1, Math.floor((event.clientY - rect.top) / rect.height * map.y.length)),
        col: Math.min(map.x.length - 1, Math.floor((event.clientX - rect.left) / rect.width * map.x.length)) })
    }}>
      <img src={transformed ? map.transformed : map.heatmap} alt={`${map.name} ${transformed ? 'scaled grayscale matrix' : 'heatmap'}`} />
    </div>
    <div className={`feature-scale ${transformed ? 'grayscale' : map.palette}`} />
    <div className="feature-scale-labels"><span>{map.range[0]}</span><span>{map.range[1]}</span></div>
    <p className="feature-values" aria-live="polite">{cell
      ? `x ${map.x[cell.col]}, y ${map.y[cell.row]}: ${map.matrix[cell.row]?.[cell.col] ?? 'outside ROI / undefined'}`
      : map.stats ? `Min ${map.stats.min.toFixed(2)} · Mean ${map.stats.mean.toFixed(2)} · Max ${map.stats.max.toFixed(2)}` : 'No valid pixels'}</p>
    <details><summary>Sampled matrix ({map.x.length} × {map.y.length})</summary>
      <div className="feature-matrix"><table><caption>{map.name}: sampled full-resolution values; — means excluded</caption><tbody>
        {map.matrix.map((row, index) => <tr key={index}>{row.map((value, col) => <td key={col} title={`x ${map.x[col]}, y ${map.y[index]}`}>{value === null ? '—' : value.toFixed(1)}</td>)}</tr>)}
      </tbody></table></div>
    </details>
  </article>
}

function FeatureImage({ path, source }: { path: string; source: FeatureSource }) {
  const [data, setData] = useState<ImageFeatures | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [transformed, setTransformed] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    getFeatures(path, source, controller.signal).then(setData).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not extract features.')
    })
    return () => controller.abort()
  }, [path, source, attempt])
  if (error) return <div className="panel empty" role="alert">{error}<br /><button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Try again</button></div>
  if (!data) return <p className="panel empty" role="status">Extracting image features…</p>
  return <div className="panel feature-panel">
    <div className="panel-heading"><div><h3>{data.name}</h3><p className="preview-caption">{data.width} × {data.height} · {data.plate_pixels.toLocaleString()} plate pixels</p></div>
      <div className="view-switch"><button aria-pressed={!transformed} onClick={() => setTransformed(false)}>Heatmaps</button><button aria-pressed={transformed} onClick={() => setTransformed(true)}>Grayscale transforms</button></div>
    </div>
    {!data.plate_pixels && <p className="feature-note" role="status">No plate found. Feature maps and histograms exclude this image; inspect its segmentation in Preprocessing.</p>}
    <div className="feature-source-images"><figure><img src={data.image} alt={`${source} source ${data.name}`} /><figcaption>{source === 'original' ? 'Original color image' : 'Final plate: glare repair → blur → CLAHE → contrast'}</figcaption></figure>
      <figure><img src={data.mask} alt="Plate region used for statistics" /><figcaption>Shared plate region for both source modes</figcaption></figure></div>
    <p className="feature-note">Fixed color scales allow comparison between images. Hover for sampled values or expand a matrix. Grayscale transforms scale each channel to 0–255 for display only; values remain in feature units. Black outside the plate is excluded. LAB is shown as separate channels, not as an RGB image.</p>
    <div className="feature-map-grid">{data.maps.map((map) => <Heatmap key={`${map.name}:${transformed}`} map={map} transformed={transformed} />)}</div>
    <HOGView data={data.hog} />
    <LBPView data={data} />
    <p className="feature-note">Sobel uses a 3×3 kernel scaled by 1/8. X points right and Y points down; direction is atan2(Y, X) in degrees (clockwise from right). Direction is undefined for zero gradients. A one-pixel inset excludes the artificial cutout boundary.</p>
  </div>
}

function HistogramPlot({ data, channel, hidden, region }: { data: FeatureHistograms; channel: HistogramChannel; hidden: string[]; region?: number }) {
  const [hover, setHover] = useState<number | null>(null)
  const edges = data.edges[channel]
  const bins = edges.length - 1
  const visible = data.groups.filter((group) => !hidden.includes(group.name))
  const valuesFor = (group: FeatureHistograms["groups"][number]) => region === undefined ? group.histograms[channel] : group.lbp_regions[region].histogram
  const max = channel === 'lbp' ? 1 : Math.max(0.001, ...visible.flatMap(valuesFor))
  const x = (i: number) => 54 + (i + 0.5) / bins * 520
  const y = (value: number) => 220 - value / max * 175
  return <article className="lab-chart"><h3>{region === undefined ? channelNames[channel] : `Row ${Math.floor(region / 4) + 1}, column ${region % 4 + 1}`}</h3>
    <svg viewBox="0 0 600 275" role="img" aria-label={`${channelNames[channel]} normalized histograms by class`} onMouseLeave={() => setHover(null)} onMouseMove={(event) => {
      const bounds = event.currentTarget.getBoundingClientRect()
      setHover(Math.max(0, Math.min(bins - 1, Math.floor(((event.clientX - bounds.left) / bounds.width * 600 - 54) / 520 * bins))))
    }}>
      {[0, 0.5, 1].map((fraction) => <g key={fraction}><line x1="54" x2="574" y1={y(max * fraction)} y2={y(max * fraction)} stroke="#e2e8e4" /><text x="48" y={y(max * fraction) + 4} textAnchor="end">{(max * fraction * 100).toFixed(1)}%</text></g>)}
      {channel === 'lbp' ? Array.from({ length: 10 }, (_, i) => <text key={i} x={x(i)} y="242" textAnchor="middle">{i}</text>) : [0, .25, .5, .75, 1].map((fraction) => <text key={fraction} x={54 + fraction * 520} y="242" textAnchor="middle">{(edges[0] + fraction * (edges[bins] - edges[0])).toFixed(channel.startsWith('frangi_') ? 2 : 0)}</text>)}
      <text x="314" y="264" textAnchor="middle">{channelUnits[channel]}</text><text x="54" y="24">{channel === 'hog' ? 'Normalized block weight (%)' : channel === 'orientation' ? 'Gradient strength per bin (%)' : 'Eligible pixels per bin (%)'}</text>
      {data.groups.map((group, index) => !hidden.includes(group.name) && <path key={group.name} d={valuesFor(group).map((value, i) => `${i ? 'L' : 'M'}${x(i)},${y(value)}`).join(' ')} stroke={colors[index % colors.length]} strokeWidth={group.name === 'good' ? 3 : 2} fill="none"><title>{group.name}: {group.images} images</title></path>)}
      {hover !== null && <line x1={x(hover)} x2={x(hover)} y1="40" y2="220" stroke="#687b70" strokeDasharray="4 4" />}
    </svg>
    <p className="histogram-hover">{hover === null ? 'Hover to compare bin values.' : `${channel === 'lbp' ? `Code ${hover}` : `${edges[hover].toFixed(1)}–${edges[hover + 1].toFixed(1)}`}: ${visible.map((group) => `${group.name} ${(valuesFor(group)[hover] * 100).toFixed(2)}%`).join(' · ')}`}</p>
  </article>
}

function ClassHistograms({ source, split }: { source: FeatureSource; split: FeatureSplit }) {
  const [data, setData] = useState<FeatureHistograms | null>(null)
  const [error, setError] = useState('')
  const [hidden, setHidden] = useState<string[]>([])
  const [regional, setRegional] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    getFeatureHistograms(source, split, controller.signal).then(setData).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not compute histograms.')
    })
    return () => controller.abort()
  }, [source, split, attempt])
  if (error) return <div className="empty" role="alert">{error}<br /><button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Try again</button></div>
  if (!data) return <p className="empty" role="status">Computing class histograms across images… The first request processes the selected dataset.</p>
  return <>
    <div className="histogram-legend">{data.groups.map((group, index) => <label key={group.name}><input type="checkbox" checked={!hidden.includes(group.name)} onChange={() => setHidden((current) => current.includes(group.name) ? current.filter((name) => name !== group.name) : [...current, group.name])} /><span style={{ background: colors[index % colors.length] }} />{group.name} · {group.images} images · {group.pixels.toLocaleString()} pixels</label>)}</div>
    {!data.groups.some((group) => group.images) ? <p className="empty">No usable images in this split.</p> : <>
      <h3 className="distribution-heading">LAB distributions</h3>
      <div className="lab-charts">{(['L', 'a', 'b'] as const).map((channel) => <HistogramPlot key={channel} channel={channel} data={data} hidden={hidden} />)}</div>
      <h3 className="distribution-heading">LBP texture distributions</h3>
      <p className="feature-note">Whole-plate and regional curves pool valid neighborhoods per class, then normalize each curve. Code 9 is non-uniform. Regional histograms use each plate’s bounding box without alignment; compare spatial cells cautiously when plates rotate. Empty cells contribute no pixels.</p>
      <div className="view-switch feature-note"><button aria-pressed={!regional} onClick={() => setRegional(false)}>Whole-plate LBP</button><button aria-pressed={regional} onClick={() => setRegional(true)}>4×4 regional LBP</button></div>
      {regional ? <div className="lbp-grid-scroll"><div className="lbp-region-grid lbp-class-grid">{Array.from({ length: 16 }, (_, region) => <HistogramPlot key={region} channel="lbp" region={region} data={data} hidden={hidden} />)}</div></div> : <div className="lab-charts"><HistogramPlot channel="lbp" data={data} hidden={hidden} /></div>}
      <h3 className="distribution-heading">HSV distributions</h3>
      <p className="feature-note">Hue wraps around: 0° and 360° are the same red. Hue excludes near-gray and near-black pixels (saturation or value below 5%); saturation and value include all plate pixels. Each channel is normalized by its own eligible count.</p>
      <div className="lab-charts">{(['H', 'S', 'V'] as const).map((channel) => <HistogramPlot key={channel} channel={channel} data={data} hidden={hidden} />)}</div>
      <h3 className="distribution-heading">HOG distributions</h3>
      <p className="feature-note">These curves pool locally normalized HOG block weights per class. Whole-plate previews use a 4×4 cell grid over each plate; models use a separate 4×4 cell grid within each patch. This pooled histogram summarizes orientation, while the full 324-value descriptor retains cell and block locations.</p>
      <div className="lab-charts"><HistogramPlot channel="hog" data={data} hidden={hidden} /></div>
      <h3 className="distribution-heading">Frangi ridge distributions</h3>
      <p className="feature-note">Dark and bright line-like structures at scales 1, 2, and 3 pixels. Responses use fixed beta 0.5 and gamma 0.05 on grayscale intensities in [0, 1]. A 12-pixel inset excludes the cutout boundary; these are ridge strengths, not defect probabilities.</p>
      <div className="lab-charts">{(['frangi_dark', 'frangi_bright'] as const).map((channel) => <HistogramPlot key={channel} channel={channel} data={data} hidden={hidden} />)}</div>
      <h3 className="distribution-heading">Sobel distributions</h3>
      <p className="feature-note">Magnitude shows the balance of flat texture and strong edges. Orientation weights each gradient by its magnitude and folds opposite directions into 0–180°, so light-to-dark and dark-to-light edges align. These are gradient normals, perpendicular to the edge itself. Zero-strength gradients do not vote. Plate rotation changes orientation peaks; global pooling loses defect locations.</p>
      <div className="lab-charts sobel-charts">{(['magnitude', 'orientation'] as const).map((channel) => <HistogramPlot key={channel} channel={channel} data={data} hidden={hidden} />)}</div>
      <div className="feature-summary"><table><caption>Class summaries · Sobel uses the plate interior, excluding its one-pixel boundary</caption><thead><tr><th>Class</th><th>Mean edge strength</th><th>Strong-edge pixels (≥10)</th><th>Pixels eligible for hue</th></tr></thead><tbody>
        {data.groups.filter((group) => !hidden.includes(group.name)).map((group) => <tr key={group.name}><th>{group.name}</th><td>{group.sobel_summary.mean_magnitude?.toFixed(2) ?? '—'}</td><td>{group.sobel_summary.strong_edge_fraction == null ? '—' : `${(group.sobel_summary.strong_edge_fraction * 100).toFixed(2)}%`}</td><td>{group.totals.H.toLocaleString()} / {group.pixels.toLocaleString()}</td></tr>)}
      </tbody></table></div>
      <p className="feature-note">The ≥10 cutoff is a fixed descriptive threshold in scaled Sobel units, not a defect decision. These summaries complement the normalized curves, which alone hide absolute edge strength.</p>
    </>}
    {data.skipped.length > 0 && <details className="feature-note"><summary>{data.skipped.length} images skipped</summary><ul>{data.skipped.map((item) => <li key={item.path}>{item.path}: {item.reason}</li>)}</ul></details>}
  </>
}

const gridFeatures = ['L*', 'a*', 'b*', 'Gradient X', 'Gradient Y', 'Magnitude', 'Direction (°)', 'Frangi · dark ridges', 'Frangi · bright ridges', 'HOG', 'LBP']
type GridPreview = { maps: Record<string, { heatmap: string; transformed: string }>; empty: boolean; error?: string }

function FeatureGrid({ images, source, feature, transformed, onSelect }: {
  images: string[]; source: FeatureSource; feature: string; transformed: boolean; onSelect: (path: string) => void
}) {
  const [previews, setPreviews] = useState<Record<string, GridPreview>>({})
  useEffect(() => {
    const controller = new AbortController()
    let next = 0
    // Limit extraction concurrency and retain only preview images, not full matrices.
    const worker = async () => {
      while (!controller.signal.aborted && next < images.length) {
        const path = images[next++]
        try {
          const data = await getFeatures(path, source, controller.signal)
          if (controller.signal.aborted) return
          const maps = Object.fromEntries(data.maps.map((map) => [map.name, { heatmap: map.heatmap, transformed: map.transformed }]))
          maps.HOG = { heatmap: data.hog.image, transformed: data.hog.image }
          maps.LBP = { heatmap: data.lbp.image, transformed: data.lbp.image }
          setPreviews((current) => ({ ...current, [path]: { maps, empty: !data.plate_pixels } }))
        } catch (error) {
          if (controller.signal.aborted) return
          setPreviews((current) => ({ ...current, [path]: { maps: {}, empty: false, error: error instanceof Error ? error.message : 'Could not extract features' } }))
        }
      }
    }
    void Promise.all(Array.from({ length: 3 }, worker))
    return () => controller.abort()
  }, [images, source])
  return <section className="panel" aria-label="Feature image grid">
    <div className="panel-heading"><h3>{feature} · all images</h3><span role="status">{Object.keys(previews).length} / {images.length} processed</span></div>
    <div className="feature-map-grid">{images.map((path) => {
      const preview = previews[path]
      const map = preview?.maps[feature]
      return <button className="feature-grid-tile" key={path} onClick={() => onSelect(path)} aria-label={`Inspect ${path}`}>
        <strong>{path.split('/').pop()}</strong><span>{path.replace('metal_plate/', '').split('/').slice(0, -1).join(' / ')}</span>
        {map ? <img loading="lazy" src={transformed ? map.transformed : map.heatmap} alt={`${feature}: ${path}`} /> : <span className="empty">{preview?.error ?? (preview ? 'Feature unavailable' : 'Extracting…')}</span>}
        {preview?.empty && <span>No plate found</span>}
      </button>
    })}</div>
  </section>
}

export default function Features() {
  const [view, setView] = useState<'single' | 'grid'>('single')
  const [feature, setFeature] = useState(gridFeatures[0])
  const [gridTransformed, setGridTransformed] = useState(false)
  const [source, setSource] = useState<FeatureSource>('original')
  const [split, setSplit] = useState<FeatureSplit>('all')
  const [images, setImages] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let active = true
    Promise.all([getImages('metal_plate', 'train'), getImages('metal_plate', 'test')]).then(([train, test]) => {
      if (!active) return
      const paths = [...train.map((name) => `metal_plate/train/${name}`), ...test.map((name) => `metal_plate/test/${name}`)]
      setImages(paths)
      setSelected(paths[0] ?? '')
    }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Could not load image list.') })
    return () => { active = false }
  }, [attempt])
  useEffect(() => {
    const navigate = (event: KeyboardEvent) => {
      if (view !== 'single' || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return
      if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable="true"]')) return
      event.preventDefault()
      setSelected((current) => {
        const index = images.indexOf(current)
        return images[index + (event.key === 'ArrowLeft' ? -1 : 1)] ?? current
      })
    }
    window.addEventListener('keydown', navigate)
    return () => window.removeEventListener('keydown', navigate)
  }, [images, view])
  const index = images.indexOf(selected)
  return <section>
    <div className="page-intro features-intro"><p className="eyebrow">METAL PLATE · FEATURE EXPLORATION</p><h2>Features</h2>
      <FeaturesWriteup />
    </div>
    <div className="panel feature-toolbar"><div className="view-switch" aria-label="Feature source">{(['original', 'preprocessed'] as const).map((value) => <button key={value} aria-pressed={source === value} onClick={() => setSource(value)}>{value === 'original' ? 'Original images' : 'Preprocessed plates'}</button>)}</div>
      <div className="view-switch" aria-label="Feature view"><button aria-pressed={view === 'single'} onClick={() => setView('single')}>Single image</button><button aria-pressed={view === 'grid'} onClick={() => setView('grid')}>Grid view</button></div>
      {view === 'grid' && <><label>Feature <select value={feature} onChange={(event) => setFeature(event.target.value)}>{gridFeatures.map((name) => <option key={name}>{name}</option>)}</select></label><label>Display <select value={gridTransformed ? 'grayscale' : 'heatmap'} onChange={(event) => setGridTransformed(event.target.value === 'grayscale')}><option value="heatmap">Heatmaps</option><option value="grayscale">Grayscale transforms</option></select></label></>}
      <p>Source selection applies to image features and all class histograms.</p>
      {error ? <p role="alert">{error} <button onClick={() => { setError(''); setAttempt(attempt + 1) }}>Retry</button></p> : <label className="feature-image-select">Image <select value={selected} onChange={(event) => { setSelected(event.target.value); setView('single') }} disabled={!images.length}>{images.map((path) => <option key={path} value={path}>{path.replace('metal_plate/', '')}</option>)}</select></label>}
      {view === 'single' && <div className="view-switch"><button disabled={index <= 0} onClick={() => setSelected(images[index - 1])}>← Previous</button><button disabled={index < 0 || index >= images.length - 1} onClick={() => setSelected(images[index + 1])}>Next →</button><span>{images.length ? `${index + 1} / ${images.length}` : 'Loading images…'}</span><span>Use ← / → to switch images</span></div>}
    </div>
    {view === 'grid' && <FeatureGrid key={source} images={images} source={source} feature={feature} transformed={gridTransformed} onSelect={(path) => { setSelected(path); setView('single') }} />}
    {view === 'single' && selected && <FeatureImage key={`${selected}:${source}`} path={selected} source={source} />}
    <section className="panel feature-panel" aria-label="Class feature histograms"><div className="panel-heading"><h3>Class distributions · good vs defect types</h3><label>Images <select aria-label="Histogram image split" value={split} onChange={(event) => setSplit(event.target.value as FeatureSplit)}><option value="all">Training + test</option><option value="test">Test only</option><option value="train">Training only</option></select></label></div>
      <p className="feature-note">Each channel overlays one pooled histogram for good images and one for each defect type. Each nonempty curve sums to 100% of its eligible pixels, gradient strength, or normalized HOG weights, using identical bins. Larger plates contribute more pixels. Training + test pools all good images; use Test only for comparisons within the test split. These are whole-plate distributions, so small defects can be diluted by normal surface pixels.</p>
      <ClassHistograms key={`${source}:${split}`} source={source} split={split} />
    </section>
  </section>
}
