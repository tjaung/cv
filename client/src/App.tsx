import { useEffect, useState } from 'react'
import DatasetViewer from './DatasetViewer'
import Preprocessing from './Preprocessing'
import Features from './Features'
import Models from './Models'
import Classifiers from './Classifiers'
import CNN from './CNN'
import './App.css'

const tabs = [
  { id: 'overview', name: 'Overview' },
  { id: 'dataset', name: 'Dataset viewer' },
  { id: 'preprocessing', name: 'Preprocessing' },
  { id: 'features', name: 'Features' },
  { id: 'models', name: 'Anomaly detection' },
  { id: 'classifiers', name: 'Classifiers' },
  { id: 'cnn', name: 'CNN' },
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
        <li><a href="#models"><strong>Anomaly detection <span aria-hidden="true">→</span></strong><span>Train normal-only patch PCA, inspect feature CSVs, and evaluate anomaly scores on test plates.</span></a></li>
        <li><a href="#classifiers"><strong>Classifiers →</strong><span>Compare supervised PCA, SVM, and KNN on a mixed 80/20 split.</span></a></li>
        <li><a href="#cnn"><strong>CNN →</strong><span>Train ResNet-18 on segmented plates, compare class metrics, and inspect learned filters and predictions.</span></a></li>
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
    {tab === 'overview' ? <Overview /> : tab === 'dataset' ? <DatasetViewer /> : tab === 'preprocessing' ? <Preprocessing /> : tab === 'features' ? <Features /> : tab === 'classifiers' ? <Classifiers /> : tab === 'cnn' ? <CNN /> : <Models />}
  </main>
}

export default App
