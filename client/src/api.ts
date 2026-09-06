const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

export type ImageSets = Record<string, string[]>

export interface ImageData {
  name: string
  data_url: string
}

export interface TestGroundTruth {
  test: ImageData
  ground_truth: ImageData | null
}

async function request(path: string, signal?: AbortSignal): Promise<Response> {
  const response = await fetch(`${API_BASE_URL}${path}`, { signal })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const detail = typeof error?.detail === 'string' ? error.detail : response.statusText
    throw new Error(`API request failed (${response.status}): ${detail}`)
  }
  return response
}

const encode = encodeURIComponent

function imagePath(imageSet: string, split: string, imageName: string): string {
  // Image names include subfolders, for example "hole/000.png".
  const name = imageName.split('/').map(encode).join('/')
  return `/images/${encode(imageSet)}/${encode(split)}/${name}`
}

export async function getImageSets(): Promise<ImageSets> {
  const response = await request('/image_sets')
  const data: { image_sets: ImageSets } = await response.json()
  return data.image_sets
}

export async function getImages(imageSet: string, split: string): Promise<string[]> {
  const response = await request(`/images/${encode(imageSet)}/${encode(split)}`)
  const data: { images: string[] } = await response.json()
  return data.images
}

/** Use directly as an <img> src to display a single image. */
export function getImageUrl(imageSet: string, split: string, imageName: string): string {
  return `${API_BASE_URL}${imagePath(imageSet, split, imageName)}`
}

/** Fetch the image bytes for processing or downloading. */
export async function getImage(imageSet: string, split: string, imageName: string): Promise<Blob> {
  const response = await request(imagePath(imageSet, split, imageName))
  return response.blob()
}

export async function getTestGroundTruth(
  imageSet: string,
  defect: string,
  imageName: string,
): Promise<TestGroundTruth> {
  const response = await request(
    `/test/ground_truth/${encode(imageSet)}/${encode(defect)}/${encode(imageName)}`,
  )
  return response.json()
}

export interface PreprocessingStep {
  number: number
  name: string
  description: string
}

export async function getPreprocessingSteps(): Promise<PreprocessingStep[]> {
  const response = await request('/preprocessing/steps/')
  const data: { steps: PreprocessingStep[] } = await response.json()
  return data.steps
}

/** Paths are relative to anomaly_dataset; omitting step returns the final Canny edges. */
export async function getPreprocessedImage(
  imagePath: string,
  step?: number,
  allowBorderTouching = true,
  signal?: AbortSignal,
  thresholdOffset = 20,
  glareCutoff = 220,
  surroundingRadius = 5,
  blendWidth = 3,
  contrastFactor = 1.5,
): Promise<{ blob: Blob; plateFound: boolean; threshold: number | null }> {
  const query = new URLSearchParams({
    image_path: imagePath,
    allow_border_touching: String(allowBorderTouching),
    threshold_offset: String(thresholdOffset),
    glare_cutoff: String(glareCutoff),
    surrounding_radius: String(surroundingRadius),
    blend_width: String(blendWidth),
    contrast_factor: String(contrastFactor),
  })
  if (step !== undefined) query.set('step', String(step))
  const response = await request(`/preprocessing/pipeline/?${query}`, signal)
  const thresholdHeader = response.headers.get('X-Isodata-Threshold')
  return { blob: await response.blob(), plateFound: response.headers.get('X-Plate-Found') !== 'false',
    threshold: thresholdHeader === null ? null : Number(thresholdHeader) }
}

export type FeatureSource = 'original' | 'preprocessed'
export type FeatureSplit = 'all' | 'train' | 'test'
export type LabChannel = 'L' | 'a' | 'b'
export type HistogramChannel = LabChannel | 'H' | 'S' | 'V' | 'magnitude' | 'orientation' | 'lbp' | 'hog' | 'frangi_dark' | 'frangi_bright'
export interface FeatureMap {
  name: string
  range: [number, number]
  palette: 'sequential' | 'diverging' | 'cyclic'
  heatmap: string
  transformed: string
  matrix: (number | null)[][]
  x: number[]
  y: number[]
  stats: { min: number; max: number; mean: number } | null
}
export interface LBPDistribution { counts: number[]; pixels: number; histogram: number[] }
export interface LBPFeatures extends LBPDistribution {
  image: string
  bounds: [number, number, number, number]
  regions: (LBPDistribution & { row: number; col: number; bounds: [number, number, number, number] })[]
}
export interface ImageFeatures {
  name: string
  source: FeatureSource
  width: number
  height: number
  hog: { image: string; bounds: number[]; descriptor: number[]; cells: number[][][]; histogram: number[]; valid_pixels: number }
  lbp: LBPFeatures
  plate_pixels: number
  image: string
  mask: string
  maps: FeatureMap[]
  histograms: Record<LabChannel, { counts: number[]; edges: number[] }>
}
export interface FeatureHistograms {
  source: FeatureSource
  split: FeatureSplit
  weighting: string
  edges: Record<HistogramChannel, number[]>
  groups: { name: string; images: number; pixels: number; splits: Record<string, number>; histograms: Record<HistogramChannel, number[]>; totals: Record<HistogramChannel, number>; interior_pixels: number; lbp_regions: { pixels: number; histogram: number[] }[]; sobel_summary: { mean_magnitude: number | null; strong_edge_fraction: number | null } }[]
  skipped: { path: string; reason: string }[]
}
export async function getFeatures(imagePath: string, source: FeatureSource, signal?: AbortSignal): Promise<ImageFeatures> {
  const query = new URLSearchParams({ image_path: imagePath, source })
  return (await request(`/features/?${query}`, signal)).json()
}
export async function getFeatureHistograms(source: FeatureSource, split: FeatureSplit, signal?: AbortSignal): Promise<FeatureHistograms> {
  const query = new URLSearchParams({ source, split })
  return (await request(`/features/histograms/?${query}`, signal)).json()
}

export interface PCAPatchRecord {
  image_id: string; patch_id: number; left: number; top: number; right: number; bottom: number; pixels: number; coverage: number
}
export interface PCAPoint { image_id: string; patch_id: number; scores: number[]; error: number }
export interface PCATest {
  model_id: string; test_id: string; image_id: string; prediction: 'GOOD' | 'BAD'; plate_score: number
  patch_threshold: number; plate_threshold: number; anomalous_patches: number; membership: string
  map_min?: number
  image: string; anomaly_map: string; coverage_mask: string; map_max: number
  patches: (PCAPatchRecord & { scores: number[]; error: number; anomalous: boolean; nearest_distance: number; mahalanobis: number; nearest_patch: PCAPatchRecord })[]
}
export interface PCAEvaluation {
  images: number; tp: number; tn: number; fp: number; fn: number; recall: number | null; precision: number | null; false_positive_rate: number | null
  groups: Record<string, { images: number; flagged: number }>
  skipped: { image_id: string; reason: string }[]
  rows: { image_id: string; label: string; actual: string; prediction: string; score: number; patches: number; anomalous_patches: number }[]
}
export interface PCAReport {
  model_id: string; name: string; features: number; components: number; retained_variance: number
  training_images: number; calibration_images: number; training_patches: number; calibration_patches: number
  patch_threshold: number; plate_threshold: number
  explained_variance_ratio: number[]
  compatible?: boolean
  config: { pipeline_signature?: string; feature_set?: string; model_type?: string; kernel?: string; nu?: number; gamma?: string | number; support_vectors?: number; distance_metric?: string; patch_size: number; variance_target: number; quantile: number; source: string }
  skipped: { image_id: string; reason: string }[]
  evaluation?: PCAEvaluation | null
}
export interface PCAModel extends PCAReport {
  component_weights: number[][]
  feature_names: string[]; explained_variance_ratio: number[]; calibration_plate_errors: number[]
  training_errors: number[]; calibration_errors: number[]; training_points: PCAPoint[]; calibration_points: PCAPoint[]
}
export interface ModelJob {
  job_id: string; kind: 'classifier_curves' | 'classifiers' | 'train' | 'test' | 'evaluate' | 'sweep' | 'evaluate_all' | 'svm_grid' | 'retrain_all'; status: 'queued' | 'running' | 'complete' | 'failed'
  combination?: number; combinations?: number; completed?: number; failed?: number; workers?: number
  phase: string; done: number; total: number; error?: string
  result?: PCATest | PCAEvaluation | { model_id: string } | { model_ids: string[]; failures: { error: string }[] }
}
export interface PCAFeaturePage { total: number; columns: string[]; rows: { record: PCAPatchRecord; error: number; values: number[] }[] }
export async function getModels(): Promise<{ models: PCAReport[]; history: (PCAReport & { archived_at: string })[]; retrain_configurations: number; active_job: ModelJob | null }> {
  return (await request('/models/')).json()
}
export async function getPCAModel(modelId: string, signal?: AbortSignal): Promise<PCAModel> {
  return (await request(`/models/pca/?model_id=${encode(modelId)}`, signal)).json()
}
export async function getModelJob(jobId: string, signal?: AbortSignal): Promise<ModelJob> {
  return (await request(`/models/jobs/${encode(jobId)}`, signal)).json()
}
export async function startPCAJob(kind: ModelJob['kind'], body: object = {}, modelId?: string): Promise<{ job_id: string }> {
  const response = await fetch(`${API_BASE_URL}/models/${kind === 'retrain_all' ? 'retrain_all' : kind === 'svm_grid' ? 'one_class_svm/grid' : `pca/${kind}`}/${modelId ? `?model_id=${encode(modelId)}` : ''}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Model request failed (${response.status})`)
  }
  return response.json()
}
export async function getPCAFeaturePage(modelId: string, split: string, view: string, offset: number, testId?: string, signal?: AbortSignal): Promise<PCAFeaturePage> {
  const query = new URLSearchParams({ model_id: modelId, split, view, offset: String(offset), limit: '15' })
  if (testId) query.set('test_id', testId)
  return (await request(`/models/pca/features/?${query}`, signal)).json()
}
export function getPCADownloadUrl(modelId: string, file: string, testId?: string): string {
  const query = new URLSearchParams({ model_id: modelId, file })
  if (testId) query.set('test_id', testId)
  return `${API_BASE_URL}/models/pca/download/?${query}`
}

export function getModelHistoryUrl(modelId: string): string {
  return `${API_BASE_URL}/models/history/${encode(modelId)}`
}

export type PCAProjection = Pick<PCATest, 'model_id' | 'image_id' | 'prediction' | 'plate_score' | 'patch_threshold' | 'plate_threshold' | 'anomalous_patches' | 'patches'>
export async function getModelProjection(modelId: string, imagePath: string, signal?: AbortSignal): Promise<PCAProjection> {
  const query = new URLSearchParams({ model_id: modelId, image_path: imagePath })
  return (await request(`/models/projection/?${query}`, signal)).json()
}

export async function classifierRequest<T>(path = '', body?: object, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/classifiers/${path}`, {
    method: body ? 'POST' : 'GET', signal,
    ...(body ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Classifier request failed')
  return data
}
