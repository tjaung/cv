import { useEffect, useState } from 'react'
import DatasetViewer from './DatasetViewer'
import Preprocessing from './Preprocessing'
import Features from './Features'
import './App.css'

const tabs = [
  { id: 'overview', name: 'Overview' },
  { id: 'dataset', name: 'Dataset viewer' },
  { id: 'preprocessing', name: 'Preprocessing' },
  { id: 'features', name: 'Features' },
]

function currentTab() {
  const hash = window.location.hash.slice(1)
  return tabs.some((tab) => tab.id === hash) ? hash : 'overview'
}

function Overview() {
  return <section className="overview">
    <div className="page-intro"><h2>Explore the dataset, one step at a time.</h2><p>Inspect the source images, compare defects with their ground-truth masks, and see how classical preprocessing isolates each metal plate.</p></div>
    <div className="panel overview-content">
      <h3>Table of contents</h3>
      <ol className="contents-list">
        <li><a href="#dataset"><strong>Dataset viewer <span aria-hidden="true">→</span></strong><span>Browse folders, switch between grid and single images, and overlay ground truth.</span></a></li>
        <li><a href="#preprocessing"><strong>Preprocessing <span aria-hidden="true">→</span></strong><span>Explore training and test metal plates through normalization, thresholding, cleanup, and masking.</span></a></li>
        <li><a href="#features"><strong>Features <span aria-hidden="true">→</span></strong><span>Explore LAB and Sobel heatmaps and compare color distributions across good plates and defect types.</span></a></li>
      </ol>
      <p>The preprocessing workflow uses a blurred grayscale copy with darkened edges to find the plate. The final mask is applied to the normalized color image so surface detail remains visible. Inspect bright-region masks and try glare removal using surrounding-color fill and boundary blending.</p>
    </div>
  </section>
}

function App() {
  const [tab, setTab] = useState(currentTab)
  useEffect(() => {
    const update = () => setTab(currentTab())
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])

  return <main>
    <header><p className="eyebrow">COMPUTER VISION WORKSPACE</p><h1>Anomaly dataset</h1><p>Explore images and understand every step of preprocessing.</p></header>
    <nav className="app-tabs" aria-label="Workspace tabs">
      {tabs.map((item) => <a key={item.id} href={`#${item.id}`} aria-current={tab === item.id ? 'page' : undefined}>{item.name}</a>)}
    </nav>
    {tab === 'overview' ? <Overview /> : tab === 'dataset' ? <DatasetViewer /> : tab === 'preprocessing' ? <Preprocessing /> : <Features />}
  </main>
}

export default App
