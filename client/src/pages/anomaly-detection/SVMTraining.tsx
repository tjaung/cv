import FeatureSetSelect from './FeatureSetSelect'
import { useState } from 'react'
import { startPCAJob } from '../../api'

export default function SVMTraining({ busy, onStarted, onError }: { busy: boolean; onStarted: (id: string) => void; onError: (message: string) => void }) {
  const [featureSet, setFeatureSet] = useState('lab_sobel')
  const [kernel, setKernel] = useState('all'), [nu, setNu] = useState('all'), [gamma, setGamma] = useState('all')
  const [patch, setPatch] = useState(64), [variance, setVariance] = useState(.95), [starting, setStarting] = useState(false)
  const nus = nu === 'all' ? [.01, .05, .1] : [Number(nu)]
  const gammas = gamma === 'all' ? ['scale', .01, .1] : [gamma === 'scale' ? gamma : Number(gamma)]
  const count = nus.length * ((kernel === 'all' || kernel === 'rbf' ? gammas.length : 0) + (kernel === 'all' || kernel === 'linear' ? 1 : 0))
  const train = async () => {
    setStarting(true); onError('')
    try {
      const result = await startPCAJob('svm_grid', { feature_set: featureSet, patch_size: patch, variance_target: variance, kernels: kernel === 'all' ? ['rbf', 'linear'] : [kernel], nus, gammas })
      onStarted(result.job_id)
    } catch (error) { onError(error instanceof Error ? error.message : 'Could not train SVM grid') }
    finally { setStarting(false) }
  }
  return <section className="panel model-training svm-training"><h3>One-class SVM · parameter grid</h3>
    <p>Good plates → standardized selected patch features → PCA → one-class SVM. Each combination trains a separate normal boundary and calibrates its own 99th-percentile cutoff on held-out good images.</p>
    <div className="model-controls"><FeatureSetSelect value={featureSet} disabled={busy || starting} onChange={setFeatureSet} />
      <label>Patch size <select disabled={busy || starting} value={patch} onChange={(e) => setPatch(Number(e.target.value))}>{[32, 64, 128].map((v) => <option key={v}>{v}</option>)}</select></label>
      <label>PCA variance <select disabled={busy || starting} value={variance} onChange={(e) => setVariance(Number(e.target.value))}>{[.9, .95, .99].map((v) => <option key={v} value={v}>{v * 100}%</option>)}</select></label>
      <label>Kernel <select disabled={busy || starting} value={kernel} onChange={(e) => setKernel(e.target.value)}><option value="all">RBF + linear</option><option value="rbf">RBF</option><option value="linear">Linear</option></select></label>
      <label>Nu <select disabled={busy || starting} value={nu} onChange={(e) => setNu(e.target.value)}><option value="all">All: 0.01, 0.05, 0.1</option>{[.01, .05, .1].map((v) => <option key={v}>{v}</option>)}</select></label>
      <label>Gamma <select disabled={busy || starting || kernel === 'linear'} value={gamma} onChange={(e) => setGamma(e.target.value)}><option value="all">All: scale, 0.01, 0.1</option>{['scale', '.01', '.1'].map((v) => <option key={v}>{v}</option>)}</select></label>
      <button disabled={busy || starting} onClick={() => void train()}>Train & evaluate {count} SVM combinations</button>
    </div><details><summary>Scoring and grid search</summary><p>Nu controls the fitted boundary: it bounds the training error fraction above and support-vector fraction below. Gamma controls RBF locality; linear kernels ignore gamma, so duplicate linear runs are omitted.</p><p>Anomaly score = negative SVM decision margin; higher means less normal. Native SVM cutoff is zero. This app instead calibrates patch scores and plate maximum scores at the 99th percentile of held-out good data. Negative scores and cutoffs are valid. Nu is not the image false-alarm rate.</p><p>This is an exhaustive parameter sweep, not supervised cross-validation. All runs use the same good-image split; test results compare combinations without selecting or refitting a winner from test labels.</p></details>
  </section>
}
