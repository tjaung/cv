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

## Models tab

The comparison table is followed by model tabs (PCA initially). The PCA view
supports background training, scoring one test image, fixed-threshold test-set
evaluation, component selectors for the eigenspace plot, explained variance,
error distributions, anomaly/coverage maps, full feature tables, and CSV export.
Distances use all retained components even when the graph displays two.
New training runs preserve previous artifacts. Training is only on good images;
99th-percentile error calibration uses separate good images and never test data.

Models now supports saved experiment selection and parameter sweeps. The
training dropdowns allow one value or all values for patch size, retained
variance target, and reconstruction-distance metric. The resulting Cartesian
product is trained and evaluated through `POST /models/pca/sweep/`.

The holistic section compares every saved run, includes PCA variance profiles,
and displays a selected image's predictions, scores, thresholds, and normalized
score ratios across all runs. The lower parameter filters select a saved run
for detailed PCA/image inspection. Raw scores from different metrics have
different units; score/threshold is not a probability. Old preprocessing runs
remain inspectable, but cannot score new images until retrained.

The Models training tabs now include **One-class SVM**. Its grid controls
kernel, nu, and gamma, with fixed selectable patch size and PCA variance for
each sweep. Results join the cross-family model statistics, per-image matrix,
and PCA summaries. Family/kernel/nu/gamma filters select individual runs.
Score-minus-threshold replaces the ratio so negative SVM scores and cutoffs
remain meaningful: positive means BAD. This margin is not a probability,
and its raw units differ across runs. Anomaly maps and histograms include
negative score ranges. Per-run inspection retains arrow browsing, patch
maps, PCA coordinates, and feature CSVs.

HOG is shown in Features as gradient-orientation glyphs, a normalized
orientation histogram, the 16×9 raw cell matrix, and all 324 descriptor values.
Class curves pool normalized HOG block weights per label and respect the
original/preprocessed source selector. Models adds a Features dropdown to
both PCA and SVM training and a feature-set filter for saved runs. Options:
LAB + Sobel, LAB + HOG, or LAB + Sobel + HOG. The same choice is persisted for
inference, feature CSVs, and PCA component contributions (including HOG).

Preprocessing step 15 is now Contrast / final plate, replacing Canny edges.
The default factor is 1.5; `getPreprocessedImage` accepts an optional final
`contrastFactor` argument. The final color plate restores repaired chroma
around the enhanced luminance. All model feature sets and the Features tab's
preprocessed source use this complete output, including glare repair, plate
blur, CLAHE, and contrast. Saved runs from the previous pipeline require
retraining; they are not silently reused with the changed inputs.

“Retrain all models” uses `POST /models/retrain_all/` to train every saved
unique configuration and replace its current fitted model after successful
training and evaluation. Previous reports and image scores remain in the
Results history table, which compares old/current metrics and follows the
selected image. JSON history downloads include all retained result records.
Saved recipes keep this button usable even when all fitted models have been
cleared. The frontend does not automatically start retraining after a clear.

Comparison and history tables support sortable column headers; clicking the
same header reverses direction. Default summary order is higher defect recall,
then lower good-image false-positive rate. An optional balanced ranking uses
(recall + specificity) / 2. The per-image comparison defaults to correct
predictions first and recalculates for the selected image. Missing results
stay last in either direction. History supports these rankings too.

The Models page is now a unified comparison view. All/PCA/One-class SVM tabs
filter summary rows, image predictions, and PCA-analysis choices. One retrain
button rebuilds all 135 supported parameter combinations with all features;
there are no individual training controls or feature-set choices. The summary
ends with one sortable column per dataset class: correct/total (percentage
of evaluated images in that class). GOOD is correct for good plates and BAD
for defect images; these are binary anomaly decisions, not multiclass labels.

Individual predictions use three columns: image, sortable model results,
and PCA scatter. Clicking a model requests a read-only projection and updates
the graph; changing images also updates it. Separate PCA analysis retains
variance and component composition. Its model dropdown, previous/next buttons,
and left/right keys while focused in that section change the analyzed model.
Elsewhere, left/right keys continue browsing images. Result history remains
stored/downloadable through the API but is no longer a separate page section.
