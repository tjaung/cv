import { useState } from 'react'

export type HighlightPatch = { left: number; top: number; right: number; bottom: number; label: string; strength?: number; color?: string }

/** Original-image coordinates stay aligned while the image scales responsively. */
export default function PatchOverlay({ src, alt, patches, description }: { src: string; alt: string; patches: HighlightPatch[]; description: string }) {
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [visible, setVisible] = useState(true)
  return <div className="patch-overlay-view">
    <button aria-pressed={visible} onClick={() => setVisible(v => !v)}>{visible ? 'Hide highlighted patches' : 'Show highlighted patches'}</button>
    <div className="patch-overlay-image">
      <img src={src} alt={alt} onLoad={event => setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
      {visible && size.width > 0 && <svg viewBox={`0 0 ${size.width} ${size.height}`} role="img" aria-label={description}>
        {patches.map((p, i) => <rect key={i} x={p.left} y={p.top} width={p.right-p.left} height={p.bottom-p.top} fill={p.color ?? '#f23b35'} fillOpacity={.08 + .24 * (p.strength ?? 1)} stroke={p.color ?? '#f23b35'} strokeWidth="2" vectorEffect="non-scaling-stroke"><title>{p.label}</title></rect>)}
      </svg>}
    </div>
    <p className="feature-note">{description}</p>
  </div>
}
