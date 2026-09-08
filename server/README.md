# Anomaly Dataset API

## Server layout

`main.py` creates the FastAPI app and includes `api/router.py`. The central
router includes each feature's `router.py`, which registers that feature's
existing endpoint URLs.

```text
server/api/
  router.py
  dataset/             # Image browsing and ground truth
  preprocessing/       # Pipeline steps and previews
  features/            # Feature images and histograms
  anomaly_detection/   # PCA, one class SVM, jobs, and model lifecycle
  classifiers/         # Supervised models, evaluations, and explanations
  cnn/                 # Whole plate and patch CNN endpoints
  comparison/          # Saved model loading and timed random evaluations
  shared/              # Dataset paths, result storage, and model saving
```

Feature folders contain endpoint handlers and a router. CNN patch processing
lives beside the CNN handlers. Tests are in `server/tests/`. Data, artifacts,
and saved results retain their existing locations. Both launch commands below
remain supported.

From the repository root:

```sh
venv/bin/python -m pip install -e . -r server/requirements.txt
venv/bin/python -m uvicorn server.main:app --reload
```

Or, from inside the `server` directory:

```sh
../venv/bin/python -m pip install -e .. -r requirements.txt
../venv/bin/python -m uvicorn main:app --reload
```

Interactive API documentation: http://127.0.0.1:8000/docs

| GET endpoint | Response |
| --- | --- |
| `/image_sets` | `{"image_sets": {"bracket_black": ["ground_truth", "test", "train"], ...}}` |
| `/images/bracket_black/test` | `{"images": ["good/000.png", "hole/000.png", ...]}` |
| `/images/bracket_black/test/hole/000.png` | Original image bytes, suitable for an `<img src="...">`. |
| `/test/ground_truth/bracket_black/hole/000.png` | `test` and `ground_truth` objects, each containing `name` and `data_url`. |
| `/preprocessing/steps/` | Numbered preprocessing stages and descriptions. |
| `/preprocessing/pipeline/?image_path=metal_plate/train/good/000.png` | Final Canny edges as a PNG image. |

Image lists include subfolders; use the returned relative name in the single-image endpoint.
The paired endpoint matches `test/<defect>/<name>.png` to
`ground_truth/<defect>/<name>_mask.png`. Each `data_url` can be used directly as an
image's `src`. For `good` test images, `ground_truth` is `null`.
Missing folders, images, or expected masks return HTTP 404. Paths outside the
dataset are rejected. The dataset location is relative to the server source files.

Preprocessing paths are relative to `anomaly_dataset`. The pipeline endpoint
accepts `step=0` through `step=15`: raw, normalized, grayscale, blur, edge darkening, ISODATA,
cleanup, plate mask, segmented plate, plate grayscale, glare threshold,
surrounding-color fill, boundary blend, plate blur, CLAHE, and Canny edges. Omitting `step` returns Canny edges (15).
The segmentation path uses the blurred grayscale copy; the final mask is applied
to the normalized, unblurred color image. The contrast function remains available separately; CLAHE now runs before final Canny.

Optional query parameters: `luminance_percentage=10`, `sobel_strength=1.0`, `blur_size=5`,
`fill_holes=true`, and `allow_border_touching=true`.
Edge darkening subtracts scaled 3×3 Sobel magnitude from the blurred grayscale
image. `sobel_strength` ranges from 0 (disabled) to 1; default 1.0. A 5×5 elliptical dilation widens the edge-strength map before subtraction. Blur size must be an odd integer from 1 to 101. Cleanup fills enclosed holes,
then applies one dilation followed by one erosion with the same 9×9 elliptical kernel. `fill_holes=false`
skips hole filling; dilation and erosion still run.

Plate extraction keeps the largest foreground component, including border
connections by default, then trims the mask with three 3×3 elliptical erosions
(approximately three pixels) before applying it to the image. This avoids rejecting a plate connected to a suspension
wire or cropped by the frame. Attached wire/hook pixels may remain. Set
`allow_border_touching=false` for strict exclusion of border-connected regions;
this can produce empty masks even when cleanup contains a plate.

Responses include `X-Preprocessing-Step` and `X-Plate-Found` headers. If no plate
is found, the final output is black and `X-Plate-Found` is `false`. Invalid
parameters or unreadable images return 422; missing images return 404; paths
outside the dataset return 400. Source images are never overwritten.

Run the API checks with `venv/bin/python -m unittest discover -s server/tests -v`.

CLAHE reference: [OpenCV CLAHE documentation](https://docs.opencv.org/4.10.0/d6/db6/classcv_1_1CLAHE.html).

ISODATA tuning: `threshold_offset` (−100 to +100, default +20) is added to each
image's automatic threshold, then clipped to 0–255. The background polarity is
chosen from the automatic threshold and stays fixed as the offset changes.
For dark plates, positive values include brighter regions but can also include
background. Responses expose the applied threshold in `X-Isodata-Threshold`.
This remains global thresholding: glare with background-like intensity cannot
be reliably distinguished with one threshold. See the
[scikit-image thresholding guide](https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding_guide.html).

Step 13 blurs a grayscale copy of the blended plate using `blur_size` (default
5×5), restoring black outside its mask. Step 14 applies CLAHE (clip limit 2,
8×8 tiles) to that blurred image. Step 15 runs Canny with
thresholds 100/200 on this enhanced image, including the plate boundary.
The blended color plate is available at step 12.

Glare removal uses `glare_cutoff=220` (0–255) to select brighter plate pixels.
Step 10 displays this binary mask. Step 11 fills each connected glare region
with the per-channel median of a surrounding ring (`surrounding_radius=5`, 1–31).
All glare pixels and pixels outside the plate are excluded from sampling.
Regions without nearby donors remain unchanged. The mask is not expanded or
contour-filled; only threshold-selected pixels are replaced.
Step 12 feathers the fill inward (`blend_width=3`, 0–31); zero disables blending.
Non-glare pixels remain exactly unchanged. Wider blending may retain brightness
in narrow highlights. Bright defects can be selected too, so inspect the mask.
The CLI accepts `--glare-cutoff`, `--surrounding-radius`, and `--blend-width`.

The replacement stages live in `surrounding_color_fill.py` and
`boundary_blending.py`. Existing Telea and mask-building functions in
`inpainting.py` remain available but are no longer used in the active pipeline;
its grayscale threshold helper is still reused. The old `patch_size` query
parameter has been replaced by `surrounding_radius` and `blend_width`.

The server imports `preprocess_plate` from the editable `CV.preprocessing`
package. Algorithms live in `CV/preprocessing`, outside the server. Reinstall
the package when creating a new Python environment; source edits take effect
without reinstalling. With Uvicorn started inside `server`, use `--reload-dir ..`
to also watch changes to `CV`.

## Features

- `GET /features/?image_path=metal_plate/test/scratches/000.png&source=original`
  returns LAB and Sobel heatmaps, scaled grayscale transforms, source/mask PNG
  data URLs, full-resolution statistics, and sampled matrices with original
  x/y coordinates. `matrix_size` (8–128, default 64) limits the longest matrix
  dimension; previews are limited to 512 pixels. Feature extraction itself uses
  full-resolution images. Null matrix entries mean excluded/undefined values.
- `GET /features/histograms/?source=original&split=all&bins=64` returns separate
  normalized LAB, HSV, and Sobel histograms for good and each defect folder in `metal_plate`.
  `split` accepts `all`, `train`, or `test`; `bins` accepts 8–128.

Both endpoints accept `source=original|preprocessed`. Preprocessed means the
repaired color plate after boundary blending, before grayscale blur/CLAHE/Canny.
Both modes use the same extracted plate ROI for features/statistics, including
original mode, to exclude background consistently. Sobel excludes one extra
pixel at the mask boundary and undefined directions. Preview scales are fixed
across images; signed channels use blue-white-red and direction uses a cyclic map.

Histograms pool eligible pixels or gradient weights per class, then normalize
each channel by its own total; nonempty channels sum to one. Good includes both train/good and test/good when
split=all; defect groups use their folder names. Larger plates carry more weight.
These are whole-plate, not ground-truth defect-only distributions. Image/pixel
counts and skipped unreadable or empty-mask images are reported. Per-image
histograms are cached by resolved path, file timestamp/size, source, and bin count;
restart the server after changing preprocessing code (or use Uvicorn reload).

Run endpoint checks with `venv/bin/python -m unittest discover -s server/tests`.

The histogram endpoint also includes H/S/V, Sobel `magnitude`, and unsigned
`orientation` in `edges` and each group's `histograms`. `totals` reports the
eligible count or summed gradient weight for each channel. Hue excludes S<5%
or V<5%; Sobel uses the plate mask inset by one pixel. Each nonempty histogram
is normalized by its own total; empty channels contain zeros. `pixels` still
reports the full plate count and `interior_pixels` reports the Sobel ROI count.
`sobel_summary` contains pixel-pooled mean magnitude and the fraction with
magnitude ≥10 (3×3 Sobel scaled by 1/8). These are descriptive measurements,
not defect predictions. Rotation and whole-plate pooling affect interpretation.

LBP is also returned by both feature endpoints. `/features/` includes `lbp`:
a nearest-neighbor preview of uniform codes 0–9 scaled to grayscale, whole-plate
counts/normalized histogram/pixel count, and 16 regions with bounds and counts.
`/features/histograms/` adds the `lbp` channel (always ten categorical bins,
independent of `bins`) and each class's `lbp_regions` (16 pooled, independently
normalized histograms and pixel counts). Neighborhoods crossing background
are excluded. Regional tiles follow each image's unaligned plate bounding box.

## Models / PCA

The Models tab trains and evaluates `CV.models.PCAAnomalyDetector`.

| Endpoint | Purpose |
| --- | --- |
| `GET /models/` | Current model comparison, evaluation metrics, active job |
| `POST /models/pca/train/` | Good-only training job; JSON `patch_size=64`, `variance_target=0.95`, `seed=42` |
| `GET /models/jobs/{job_id}` | Progress, completion result, or error |
| `GET /models/pca/` | PCA metadata, variance, training/calibration coordinates and errors |
| `POST /models/pca/test/` | JSON `image_path`; patch scores, nearest references, anomaly map |
| `POST /models/pca/evaluate/` | Evaluate all metal_plate/test images with fixed thresholds |
| `GET /models/pca/features/` | Complete feature rows, paginated with `offset`, `limit` |
| `GET /models/pca/download/` | CSV downloads for training, calibration, test, components, evaluation |

Train/test/evaluate return a `job_id`. Model operations run in a single background
worker; concurrent mutations return 409. Jobs are in-memory and stop on server
restart. Completed models and test results persist under ignored
`artifacts/models/pca/<model_id>/`. A new training run preserves older runs and
atomically updates the current-model pointer. Optional `model_id` selects a run.

Feature browsing accepts `split=training|calibration|test` and
`view=raw|standardized|reconstructed|pca`. Test features require `test_id`.
Downloads use `file=training|calibration|test|components|evaluation`; tests require
`test_id`. Complete CSV includes every vector, standardized vector, standardized
reconstruction, error, and PCA coordinate. The client table displays 15 rows per
page and every feature column. Source paths are resolved within the dataset.

Training uses only train/good, approximately 80/20 split by image with identical
files grouped together. Both patch and plate thresholds are fixed to the 99th
percentile of held-out good scores. Evaluation never adjusts thresholds and
reports skipped/unscorable images separately. Defect recall, good false alarms,
precision, and per-folder outcomes appear in the comparison panel.

See [PCA feature schema and calibration](../CV/models/README.md).

### PCA experiment sweeps

`POST /models/pca/sweep/` trains and evaluates the Cartesian product of
`patch_sizes` (32, 64, 128), `variance_targets` (0.90, 0.95, 0.99), and
`distance_metrics` (`l1`, `l2`, `mahalanobis`). Omit fields to run all 27;
pass singleton lists to train one configuration. The seed defaults to 42.
Every run gets its own ID, calibrated thresholds, feature CSVs, PCA weights,
and test evaluation. Extracted features are reused during the job; every
combination uses the same split by image. Poll the returned job ID as usual;
sweep results include completed IDs and any per-combination failures.

`GET /models/` lists all saved runs with evaluations, PCA variance ratios,
and preprocessing compatibility. `POST /models/pca/evaluate_all/` evaluates
all compatible saved runs. Existing `model_id` query parameters select the
run for inspection, scoring, exports, or individual evaluation.

Metrics score the standardized reconstruction residual: L1 is its absolute
sum, L2 is its Euclidean norm, and Mahalanobis uses training-only residual
covariance plus a diagonal ridge of 1% of average residual variance (minimum
1e-8). Each metric calibrates patch and plate thresholds independently at the
99th percentile on held-out good images. Old runs without a metric retain
squared-L2 scores. The separately reported nearest training-patch distance
is still Euclidean in retained PCA space; it does not drive classification.

### One-class SVM experiments

`POST /models/one_class_svm/grid/` trains every distinct requested combination
and evaluates each saved run. Defaults are `patch_size: 64`,
`variance_target: 0.95`, `kernels: ["rbf", "linear"]`,
`nus: [0.01, 0.05, 0.1]`, `gammas: ["scale", 0.01, 0.1]`, and `seed: 42`.
Gamma is ignored for linear kernels, giving 12 default combinations.
Singleton lists train one configuration. Poll `/models/jobs/{job_id}`;
completed jobs return saved IDs and any per-combination failures.

The normal-only pipeline is segmentation → LAB/Sobel patch features →
training-fitted standardization → training-fitted PCA → one-class SVM.
Nu, kernel, and gamma control the boundary. Patch anomaly scores are negative
SVM decision margins, so higher is more anomalous. Plate scores use the
maximum patch score. Patch and plate thresholds are independently calibrated
at the 99th percentile using held-out good images; the native zero-margin
boundary is not the final calibrated app verdict. Scores/cutoffs may be
negative. This grid is an exhaustive experiment sweep, not a supervised
cross-validation procedure: it neither fits on test labels nor chooses a
winner from test performance.

`GET /models/` includes both families. SVM runs support
`GET /models/one_class_svm/`, `POST /models/one_class_svm/test/`, and
`POST /models/one_class_svm/evaluate/` with `model_id`. Existing shared
feature/download endpoints also accept SVM run IDs. For compatibility, both
families use the existing `artifacts/models/pca/{model_id}` artifact store;
`config.model_type` selects the loader. Numeric SVM support vectors,
coefficients, and intercepts are persisted without pickle. The reusable
class is `CV.models.OneClassSVMDetector`; scikit-learn is needed for training.
The PCA visualization shows the SVM inputs, not the full fitted SVM boundary.

### HOG feature sets

PCA training/sweeps and one-class SVM grids accept `feature_set`:
`lab_sobel` (105 values, existing default), `lab_hog` (366), or
`lab_sobel_hog` (429). Each HOG patch uses a 4×4 cell grid, nine unsigned
orientation bins with interpolated angular votes, and overlapping 2×2-cell
blocks normalized by L2-Hys (normalize, clip at 0.2, normalize again).
The HOG portion contains 324 values. Cell dimensions follow patch dimensions,
including partial edge patches; gradients are computed before patch slicing
and only valid plate-interior pixels vote. No patch seam gradients are added.

`CV.features.hog_features` provides a whole-ROI visualization/descriptor;
`hog_from_gradients` reuses existing Sobel maps. `CV.models.extract_model_features`
extracts the chosen model feature set. `/features/` returns `hog` with an image,
cell histogram matrix, normalized descriptor, and pooled orientation summary.
`/features/histograms/` adds HOG class curves pooling normalized block weights.
The feature explorer describes a whole-plate grid; classifiers retain a
separate grid in every patch. Orientation-bin centers are 0°, 20°, …, 160°;
orientations wrap at 180°. Preview line directions represent gradient normals.

Feature schemas and signatures are stored per run, and CSVs/PCA loadings use
that run's feature names. Legacy LAB/Sobel schemas remain loadable and their
extractor is unchanged. Adding HOG requires training a new run; it is never
appended to an existing fitted scaler or model. The shared comparison table
and model filters include the feature set.

### Final preprocessing output for features

The pipeline now ends with plate blur → CLAHE → contrast enhancement (default
factor 1.5 around the 127.5 midpoint), replacing the final segmented Canny
edges. `contrast_factor` is available on `/preprocessing/pipeline/` (1–10).
Step 15 returns the final color plate: enhanced grayscale luminance combined
with repaired YCrCb chroma, converted to BGR and masked again. Values outside
the plate stay black; gamut conversion may clip saturated colors.

`PreprocessingResult.contrast_plate` is the enhanced grayscale matrix and
`final_plate` is the final color feature input. The old `segmented_edges`
result is removed; the independent Canny function remains available.
Both PCA and SVM extract all feature sets from `final_plate`, with glare
repair enabled. Features' preprocessed mode uses exactly the same final
image; original mode remains available as an explicit comparison. This
supersedes earlier descriptions of extracting before glare repair or CLAHE.
The preprocessing signature changes, so old runs remain inspectable but must
be retrained before scoring images through this new pipeline.

### Replacing models and retaining results

`POST /models/clear/` archives all model reports, evaluations, and saved
per-image score/patch results before deleting fitted model directories.
It also persists unique training recipes in `artifacts/models/pca/recipes.json`.
The clear operation rejects requests while a model job is running.

`POST /models/retrain_all/` rebuilds the saved recipes plus any current model
configurations using the current feature/preprocessing implementation. Duplicate
recipes are trained once. The recipe includes family, seed, feature set, patch
size, retained-variance target, coverage, and PCA metric or SVM kernel/nu/gamma.
Each new run trains and evaluates before it replaces matching current runs.
Old results are archived first; failed replacements retain the current run.
Retraining does not select parameters based on test performance.

History records live in `artifacts/models/pca/history/{old_model_id}.json`.
They include hyperparameters, preprocessing signature, PCA summary statistics,
full test-set predictions and metrics, and numerical per-image/patch results.
Image blobs and fitted weights/feature-vector files are not retained in history.
`GET /models/history/{model_id}` downloads a record. `GET /models/` includes
`history` and `retrain_configurations` alongside current runs. History is not
removed by clearing or replacement and is available for cross-run comparison.

### Unified all-feature comparison workflow

All new API training requests now use `lab_sobel_hog` (429 values); requests
for feature subsets are rejected. Legacy artifacts remain loadable for
comparison. The retrain-all catalog is the full supported grid, independent
of saved recipes: patch sizes 32/64/128 × variance targets .90/.95/.99 ×
3 PCA distances (27 runs), plus the same patch/variance grid × 12 unique
SVM kernel/nu/gamma combinations (108 runs). Total: 135. Legacy feature-set
variants and squared-L2/L2 equivalents match one replacement configuration.
Historical reports retain their actual original settings.

`GET /models/projection/?model_id=...&image_path=...` returns the selected
image's patch PCA coordinates and model scores without creating a job or
writing test artifacts. It uses the same preprocessing/signature validation
as scoring. The client uses it when a prediction-table model row is clicked.


Frangi features: `CV.features.frangi_features` returns dark/bright multiscale
ridge responses (sigma 1, 2, 3 pixels; beta 0.5; fixed gamma 0.05 on [0,1]
grayscale) and a 12-pixel inset validity mask. Feature previews include both
maps and pooled class histograms. New API training uses
`lab_sobel_hog_frangi` (459 values): the existing 429 values plus a normalized
12-bin histogram and mean/std/max for each polarity per patch. Empty Frangi
interiors produce zero descriptors. Existing saved models retain their original
schema; retrain all to include Frangi. No training is started automatically.

Retrain-all runs up to three concurrent workers, grouped by patch size, with a
shared-within-worker feature cache and one BLAS thread per operation. Finished
models appear in the client every five seconds. Failed saves remove their new
incomplete directory; disk-full errors stop scheduling further configurations.
Previously incomplete folders are left untouched.


## Supervised classifiers

`POST /classifiers/train/` with `{"seed":42}` runs 42 PCA-centroid/SVC/KNN
configurations. All use a shared stratified mixed 80/20 split and up to five
grouped CV folds on training data only. Exact duplicate content stays together.
Scaling and PCA are fitted within each fold; final models refit on the full 80%.
Default ranking is CV macro F1. Every model predicts original class-folder labels.

`GET /classifiers/` returns split membership, reports, failures and the active job.
Poll `/models/jobs/{job_id}` for progress. `GET /classifiers/inspect/` takes
`model_id` and `image_index` (the index in the shared holdout). It returns every
training point and only the selected holdout point, actual KNN neighbors, SVM
support-vector indices, and a stable PC1/PC2 projection. Fixed display bounds
cover the training points and all saved holdout projections, while only the
selected holdout point is drawn. Class-colored decision regions use a fixed slice through the training mean
in the remaining PCs; they stay unchanged when switching selected images.

Runs live in `artifacts/classifiers/<run_id>/`, with shared feature CSV/NPZ,
split and CV fold manifests, fitted joblib files, and atomic summary updates.
The current pointer selects the latest run; earlier runs stay saved. Extraction
failures stop the run rather than silently changing the shared holdout.

`GET /classifiers/review/?run_id=<id>&image_index=<n>` classifies an image
with every fitted classifier in that saved run. Indices address the combined
`summary.train + summary.test` inventory. Results include original class,
training/holdout membership, prediction, and correctness for every model.
It reuses the run's saved feature vectors without retraining or writing files.
The classifier listing includes `summary.run_id` for this purpose.

The client random-review section selects unused paths without replacement and
persists a path-to-boolean dictionary, current selection, and predictions in
browser local storage per run/model set. An image is counted only after all
models return successfully; retries keep the same image. Arrow navigation is
scoped to the focused random-review section. At exhaustion, the final ranking
shows overall, training, holdout, and per-class correct counts. Results can be
downloaded as JSON. Combined accuracy includes seen training images and is not
an estimate of unseen-data performance.

Classifier overfitting diagnostics include final-model training evaluation next
to holdout evaluation, and training-minus-holdout accuracy/macro-F1 gaps in
percentage points. CV metrics remain validation-fold scores, not training scores.
`GET /classifiers/training_metrics/?run_id=<id>` returns evaluations keyed by
model ID. New models save these during training; older runs are evaluated from
saved training vectors and fitted models, cached in memory, without retraining.
Training scores can be optimistic, particularly KNN self-matches. A positive
gap is a diagnostic signal, not a definitive overfitting classification.

Learning curves are available per selected classifier via
`GET /classifiers/learning_curves/?run_id=...&model_id=...` and generated with
`POST /classifiers/learning_curves/` (`run_id`, `model_id` JSON body). Generation
uses the shared background-job queue. Each point refits an unfitted pipeline
copy on nested 20/40/60/80/100% class-stratified training-group subsets in the
saved CV folds. It reports training/validation classification error (1-accuracy)
and balanced error (1-balanced accuracy), with fold standard deviations.
These are sample-size learning curves, not optimizer/epoch loss histories.
No held-out test data is used and fitted model artifacts are not replaced.
Points with any failed fold are marked unavailable. Results are saved beside
the selected model as `<model_id>-learning-curves.json`.

Patch highlights are available in individual predictions. Anomaly Detection
uses the selected model's actual `patches[].anomalous` flags and scores; its
patch threshold is distinct from the image-level decision threshold.

`GET /classifiers/patch_explanation/` takes `run_id`, `model_id`, and
`image_index` in the combined train+holdout inventory. It verifies that current
extracted features match the saved run, omits one patch feature vector at a
time, recomputes pooled mean/std, and measures the drop in predicted-class
support. Support is a decision score for PCA/SVM, or neighbor vote share for
KNN. The client highlights up to five positive contributions for predicted
defect classes. This is a feature-omission explanation, not local defect
classification, segmentation, or ground truth. It is computed on request and
does not refit or overwrite the classifier. The random-review prediction table
also selects which model's patch explanation to inspect.

### CNN / ResNet-18

The CNN tab uses these routes:

- `GET /cnn/`: trained runs, saved train/test metrics and active job.
- `POST /cnn/train/`: `{ "epochs": 5, "batch_size": 16, "learning_rate": 0.0001, "seed": 42 }`.
- `POST /cnn/test/`: `{ "run_id": "..." }`; evaluates the saved 20% holdout.
- `GET /cnn/weights/?run_id=...`: first convolution filters, final classifier
  weights and per-layer weight statistics.
- `GET /cnn/predictions/?run_id=...`: held-out predictions and softmax scores.
- `GET /cnn/inspect/?run_id=...&image_index=0`: exact transformed segmented
  input, class activation overlay, 512 pooled features and eight active channels.
- `POST /cnn/save/`: `{ "run_id": "..." }`; retains the selected checkpoint.

Train/test requests return a job ID; poll `/models/jobs/{job_id}`. Training uses
ImageNet weights (downloaded on first use), fine-tunes all layers, and evaluates
training metrics without using holdout data. Testing uses the saved split and
checks image hashes and preprocessing compatibility. The torch dependencies are
optional: `pip install -e '.[cnn]'` from the project root.

Results and CSVs live in `CV/results/data/cnn/<run_id>/`; temporary checkpoints
live in `artifacts/cnn/<run_id>/`. Save model copies a checkpoint to
`CV/results/models/cnn/<run_id>/`. Test image/activation snapshots and learned
weight views remain usable without checkpoints. Re-testing requires a retained
checkpoint and unchanged inputs. No training starts just by opening the tab.

CNN training also accepts `"variant": "defect_weighted"` (default: `"standard"`).
This creates a separate ResNet run with good=1 and each defect=2 cross-entropy
weights. The summary/checkpoint stores the variant and ordered class weights.
Evaluation metrics remain unweighted; the training loss curve is weighted for
this variant. The CNN tab's variant selector chooses which version to train.

CNN `variant: "patch"` trains a binary patch ResNet with `patch_size: 64`
(default, 16–256), half-patch stride and 50% minimum plate coverage. It uses
pixel masks for patch labels after the image-level train/test split. Missing
annotation masks are errors. The same train/test/inspection endpoints return
binary image metrics, `patch_metrics`, and cached localized patch overlays.
Inspection responses include `patches` with coordinates, target, score and
anomalous flag. Image classification uses max patch score >= 0.5. These binary
metrics should not be interpreted as four-class defect-type classification.

CNN `variant: "patch_multiclass"` adds four-class patch training with the same
patch settings. Ground-truth-positive patches inherit the image's defect type;
clean regions are good. Patch predictions use argmax. With at most two bad patches, the image is good
only if mean good probability across all patches is strictly highest. Otherwise
the strongest non-good patch decides (good if all patches predict good).
Inspection patch rows include `prediction`, `actual`, and four `probabilities`.

CNN training accepts `num_workers` (0–8, default 2). The CNN tab exposes this as
Data workers. Each job uses a temporary segmented-plate/mask cache and spawned
persistent data-loading workers. Caches are shared through read-only memory maps,
then workers are stopped and temporary files removed on success or failure.
Testing uses the run's stored worker setting (2 for older runs). Set 0 for a
synchronous loader if the local environment cannot start worker processes.

The preprocessing viewer has separate `segmentation` and `full` pipelines.
`GET /preprocessing/steps/?pipeline=segmentation` returns stages 0–9, ending
with a CNN patch-extraction preview (64×64, stride 32, ≥50% plate coverage).
`pipeline=full` returns the full stages 0–15. Pass the same `pipeline` to
`/preprocessing/pipeline/`; omitting `step` selects that pipeline's final stage.
The default pipeline remains `full` for existing clients. Segmentation requests
never run glare repair, CLAHE or the final contrast stage. Patch boxes indicate
input regions, not model predictions.
