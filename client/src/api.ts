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
