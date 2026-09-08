import { useEffect, useState } from 'react'
import DatasetViewer from './pages/dataset/DatasetViewer'
import Preprocessing from './pages/preprocessing/Preprocessing'
import Features from './pages/features/Features'
import Models from './pages/anomaly-detection/Models'
import Classifiers from './pages/classifiers/Classifiers'
import CNN from './pages/cnn/CNN'
import Comparison from './pages/comparison/Comparison'
import Overview from './pages/overview/Overview'
import './App.css'

const tabs = [
  { id: 'overview', name: 'Overview' },
  { id: 'dataset', name: 'Dataset viewer' },
  { id: 'preprocessing', name: 'Preprocessing' },
  { id: 'features', name: 'Features' },
  { id: 'models', name: 'Anomaly detection' },
  { id: 'classifiers', name: 'Classifiers' },
  { id: 'cnn', name: 'CNN' },
  { id: 'comparison', name: 'Comparison and Business Use' },
]

function currentTab() {
  const hash = window.location.hash.slice(1)
  return tabs.some((tab) => tab.id === hash) ? hash : 'overview'
}


function App() {
  const [tab, setTab] = useState(currentTab)
  useEffect(() => {
    const update = () => setTab(currentTab())
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])

  return <main>
    <header><p className="eyebrow">Tristar AI Assessment</p><p className="eyebrow">Tim Jaung</p><h1>MPDD Defect Dataset</h1></header>
    <nav className="app-tabs" aria-label="Workspace tabs">
      {tabs.map((item) => <a key={item.id} href={`#${item.id}`} aria-current={tab === item.id ? 'page' : undefined}>{item.name}</a>)}
    </nav>
    {tab === 'overview' ? <Overview /> : tab === 'dataset' ? <DatasetViewer /> : tab === 'preprocessing' ? <Preprocessing /> : tab === 'features' ? <Features /> : tab === 'classifiers' ? <Classifiers /> : tab === 'cnn' ? <CNN /> : tab === 'comparison' ? <Comparison /> : <Models />}
  </main>
}

export default App
