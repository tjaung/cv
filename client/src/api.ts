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
): Promise<{ blob: Blob; plateFound: boolean; threshold: number | null }> {
  const query = new URLSearchParams({
    image_path: imagePath,
    allow_border_touching: String(allowBorderTouching),
    threshold_offset: String(thresholdOffset),
    glare_cutoff: String(glareCutoff),
    surrounding_radius: String(surroundingRadius),
    blend_width: String(blendWidth),
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
export type HistogramChannel = LabChannel | 'H' | 'S' | 'V' | 'magnitude' | 'orientation' | 'lbp'
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
