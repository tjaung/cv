# Anomaly Dataset API

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
