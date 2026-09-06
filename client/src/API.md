# Dataset API

Start FastAPI on port 8000 and run the client with `pnpm dev`.
Vite forwards `/api` requests to FastAPI during development.

```tsx
import { getImageSets, getImages, getImageUrl, getImage, getTestGroundTruth } from './api'

const sets = await getImageSets() // { bracket_black: ['ground_truth', 'test', 'train'], ... }
const names = await getImages('bracket_black', 'test') // ['good/000.png', 'hole/000.png', ...]
const src = getImageUrl('bracket_black', 'test', 'hole/000.png')
const blob = await getImage('bracket_black', 'test', 'hole/000.png')
const pair = await getTestGroundTruth('bracket_black', 'hole', '000.png')
// Display pair.test.data_url and pair.ground_truth?.data_url as image src values.
// ground_truth is null for good images. Failed requests throw an Error.
```

For production, configure your web server to forward `/api` to FastAPI, stripping
the `/api` prefix, or set `VITE_API_BASE_URL` at build time to your server URL.
An API hosted on a different origin needs CORS configured on the server.

## Preprocessing

```ts
import { getPreprocessingSteps, getPreprocessedImage } from './api'

const steps = await getPreprocessingSteps()
const { blob, plateFound } = await getPreprocessedImage('metal_plate/train/good/000.png')
// Optional arguments: step (0–15; omitted means Canny edges), allowBorderTouching, AbortSignal.
// Create an object URL for display and revoke it when it is no longer needed.
```

The app opens on Overview, with links to `#dataset` and `#preprocessing`.
Preprocessing includes metal-plate training and test images, split filters, and
numbered stage buttons in grid and single-image views. Stage selection persists
when opening images, navigating with Left/Right, and returning to the grid.
Thumbnails are processed as they approach the visible part of the grid; requests
are cancelled when the view changes. Border-connected regions are allowed by default; uncheck the option for strict border exclusion.
When using a different API origin, expose `X-Plate-Found` in your CORS configuration
so the client can display the no-plate indicator.

The displayed steps are Raw → Normalize → Grayscale → Blur → Edge darkening → ISODATA → Cleanup →
Plate mask → Segmented plate → Plate grayscale → Glare threshold → Surrounding color fill →
Boundary blend → Plate blur → CLAHE → Canny edges. ISODATA receives the edge-darkened grayscale image. Cleanup
fills holes, dilates once, and erodes once; the final mask is applied to the normalized color
image. The contrast utility is retained outside the active pipeline. Plate blur uses the same blur size as the initial blur (default 5×5).

Preprocessing uses fixed client defaults: ISODATA offset +20, glare cutoff 220,
surrounding ring radius 5, and inward blend width 3. There are no tuning sliders.
The API helper still accepts these parameters for programmatic use and returns
`threshold` alongside the Blob and plate status. Cross-origin APIs must expose
`X-Isodata-Threshold` through CORS to show the threshold readout.

Dataset comparisons use a show/hide checkbox with a 50% ground-truth overlay
in grid and single-image views.

## Features tab

`getFeatures(imagePath, source, signal?)` requests LAB and Sobel maps and sampled
matrix values. `getFeatureHistograms(source, split, signal?)` requests normalized
class distributions. The source switch updates both sections. Preprocessed
features use the repaired color plate; both modes exclude the same background.
The tab provides heatmap/grayscale views, matrix inspection, previous/next image
selection, and three overlaid LAB plots with class visibility checkboxes.
Use Test only to avoid pooling training good images with test defect images.

Class comparison now includes LAB, HSV, and Sobel magnitude/orientation plots
under one shared legend, source toggle, and split selection. The summary table
shows mean Sobel strength, strong-edge fraction (≥10), and hue-eligible pixels.
Orientation is weighted by magnitude and unsigned (0–180°); hue is circular
(0–360°) and excludes near-gray/near-black pixels. Empty channels have zero curves.

LBP adds a code image with a labeled bounding-box grid, one whole-plate histogram,
and 16 regional histograms for the selected image. Clicking a regional card
highlights its position. Class comparisons can switch between whole-plate and
4×4 regional LBP overlays using the existing class/source/split controls. Every
LBP plot uses the same ten categories and 0–100% vertical scale. Narrow screens
scroll the regional grids horizontally to preserve their 4×4 spatial layout.
