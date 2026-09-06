import { useState } from 'react'
import type { ImageFeatures, LBPDistribution } from './api'

function Bars({ data, title }: { data: LBPDistribution; title: string }) {
  return <div className="lbp-bars">{title && <h3>{title}</h3>}
    {data.pixels ? <svg viewBox="0 0 300 160" role="img" aria-label={`${title} normalized LBP histogram`}>
      {[0, .5, 1].map((value) => <g key={value}><line x1="32" x2="292" y1={125 - 100 * value} y2={125 - 100 * value} stroke="#e2e8e4" /><text x="29" y={129 - 100 * value} textAnchor="end">{value * 100}%</text></g>)}
      {data.histogram.map((value, index) => <g key={index}><rect x={35 + index * 25.5} y={125 - value * 100} width="20" height={value * 100} fill={index === 9 ? '#a555a6' : '#278568'}><title>{index === 9 ? 'Non-uniform' : `Code ${index}`}: {(100 * value).toFixed(2)}% · {data.counts[index]} pixels</title></rect><text x={45 + index * 25.5} y="142" textAnchor="middle">{index}</text></g>)}
    </svg> : <p className="empty">No valid neighborhoods</p>}
    <p>{data.pixels.toLocaleString()} valid pixels</p>
  </div>
}

export default function LBPView({ data }: { data: ImageFeatures }) {
  const [selected, setSelected] = useState<number | null>(null)
  const texture = data.lbp
  return <section className="lbp-section" aria-label="Selected image LBP analysis">
    <h3>Local binary patterns · {data.name}</h3>
    <p className="feature-note">Uniform LBP, 8 neighbors, radius 1 pixel. Codes 0–8 count set bits in uniform patterns; 9 groups non-uniform patterns. Codes describe local patterns, not edge strength. Each histogram uses the same ten bins and a 0–100% scale.</p>
    <div className="lbp-overview">
      <figure className="lbp-image"><svg viewBox={`0 0 ${data.width} ${data.height}`} role="img" aria-label="LBP image with 4 by 4 plate bounding-box grid">
        <image href={texture.image} width={data.width} height={data.height} />
        {texture.regions.map((region, index) => <g key={index}>
          <rect x={region.bounds[0]} y={region.bounds[1]} width={region.bounds[2] - region.bounds[0]} height={region.bounds[3] - region.bounds[1]} fill={selected === index ? '#f8dd7950' : 'none'} stroke="#ffe8a0" strokeWidth={selected === index ? 4 : 1.5} />
          <text x={region.bounds[0] + 5} y={region.bounds[1] + 22} fill="#fff4c7" stroke="#000" strokeWidth=".7" fontSize="20">{region.row + 1},{region.col + 1}</text>
        </g>)}
      </svg><figcaption>LBP image (codes scaled to grayscale) with row,column labels</figcaption></figure>
      <Bars data={texture} title="Whole plate" />
    </div>
    <p className="feature-note">The grid divides the extracted plate’s bounding box. All neighborhood codes are computed before splitting, so tile boundaries do not create edges. Neighborhoods touching background are excluded. Click a cell below to highlight it. Plates are not aligned; corresponding cells can cover different physical areas across rotated images.</p>
    <div className="lbp-grid-scroll"><div className="lbp-region-grid">{texture.regions.map((region, index) => <div key={index} className={`lbp-region ${selected === index ? 'selected' : ''}`}>
      <button aria-pressed={selected === index} onClick={() => setSelected(selected === index ? null : index)}>Row {region.row + 1}, column {region.col + 1}</button>
      <Bars data={region} title="" />
    </div>)}</div></div>
  </section>
}
