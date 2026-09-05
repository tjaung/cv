"""HTTP access to the same preprocessing pipeline used by the local demo."""

from typing import Annotated

import cv2
from fastapi import HTTPException, Query, Response

from .os_helpers import image_path as resolve_image_path

from CV.preprocessing import preprocess_plate


STEPS = [
    {'number': 0, 'name': 'Raw image', 'description': 'The original image before preprocessing.'},
    {'number': 1, 'name': 'Normalize', 'description': 'Move each pixel’s luminance toward the midpoint, retaining color.'},
    {'number': 2, 'name': 'Grayscale', 'description': 'Convert the normalized color image to grayscale.'},
    {'number': 3, 'name': 'Blur', 'description': 'Create a slightly blurred grayscale copy using a 5×5 Gaussian kernel by default.'},
    {'number': 4, 'name': 'Edge darkening', 'description': 'Widen the Sobel edge-strength map with a 5×5 dilation, then darken edges at full strength for thicker, darker outlines.'},
    {'number': 5, 'name': 'ISODATA', 'description': 'Threshold the edge-darkened grayscale image and identify background from the border.'},
    {'number': 6, 'name': 'Cleanup', 'description': 'Fill enclosed holes, then dilate once and erode once using the same 9×9 elliptical kernel.'},
    {'number': 7, 'name': 'Plate mask', 'description': 'Keep the largest foreground region, including border connections by default, fill holes, then erode three times with a 3×3 elliptical kernel to trim the thin background border.'},
    {'number': 8, 'name': 'Segmented plate', 'description': 'Apply the mask to the normalized, unblurred color image, preserving surface detail.'},
    {'number': 9, 'name': 'Plate grayscale', 'description': 'Convert the segmented color plate to grayscale for glare detection. Starts from the segmented color plate.'},
    {'number': 10, 'name': 'Glare threshold', 'description': 'Select pixels brighter than the glare cutoff (default 220), inside the plate only. White marks highlights to repair.'},
    {'number': 11, 'name': 'Surrounding color fill', 'description': 'Fill each glare region with the median BGR color of a surrounding ring (radius 5 by default), excluding other glare and the background. Regions with no nearby valid donors remain unchanged.'},
    {'number': 12, 'name': 'Boundary blend', 'description': 'Blend the color fill inward over 3 pixels by default. Only glare pixels change; narrow highlights may retain some brightness. Set blend width to zero for a solid fill.'},
    {'number': 13, 'name': 'Plate blur', 'description': 'Apply Gaussian blur to a grayscale copy of the blended plate (5×5 by default), keeping the background black.'},
    {'number': 14, 'name': 'CLAHE', 'description': 'Enhance local contrast on the blurred grayscale plate (clip limit 2, 8×8 tiles), keeping the background black.'},
    {'number': 15, 'name': 'Canny edges', 'description': 'Run Canny (100/200) on the CLAHE result as the final step. Includes the plate boundary against the masked background.'},
]
LAST_STEP = len(STEPS) - 1
COLOR_FILL_STEP = 11


def get_preprocessing_steps():
    return {'steps': STEPS}


def run_preprocessing_pipeline(
    image_path: Annotated[str, Query(min_length=1, description='Path relative to anomaly_dataset, e.g. metal_plate/train/good/000.png')],
    step: Annotated[int, Query(ge=0, le=LAST_STEP)] = LAST_STEP,
    luminance_percentage: Annotated[float, Query(ge=0, le=100)] = 10,
    blur_size: Annotated[int, Query(ge=1, le=101)] = 5,
    sobel_strength: Annotated[float, Query(ge=0, le=1)] = 1.0,
    threshold_offset: Annotated[float, Query(ge=-100, le=100)] = 20,
    glare_cutoff: Annotated[float, Query(ge=0, le=255)] = 220,
    surrounding_radius: Annotated[int, Query(ge=1, le=31)] = 5,
    blend_width: Annotated[int, Query(ge=0, le=31)] = 3,
    fill_holes: bool = True,
    allow_border_touching: bool = True,
):
    """Return the final Canny edges as PNG, or an intermediate cumulative stage."""
    if blur_size % 2 == 0:
        raise HTTPException(422, 'Blur size must be odd')
    path = resolve_image_path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(422, 'Could not decode image')
    result = preprocess_plate(
        image, glare_cutoff=glare_cutoff, surrounding_radius=surrounding_radius, blend_width=blend_width, run_glare_fill=step >= COLOR_FILL_STEP,
        luminance_percentage=luminance_percentage, blur_size=blur_size,
        fill_holes=fill_holes, allow_border_touching=allow_border_touching, sobel_strength=sobel_strength, threshold_offset=threshold_offset,
    )
    stages = [image, result.normalized, result.gray, result.blurred, result.edge_bold,
              result.threshold_mask, result.cleaned_mask, result.plate_mask, result.plate,
              result.plate_gray, result.glare_mask, result.color_filled_plate, result.blended_plate, result.blurred_plate, result.clahe_plate, result.segmented_edges]
    success, encoded = cv2.imencode('.png', stages[step])
    if not success:
        raise HTTPException(500, 'Could not encode preprocessing result')
    return Response(encoded.tobytes(), media_type='image/png', headers={
        'X-Preprocessing-Step': str(step),
        'X-Isodata-Threshold': str(result.threshold),
        'X-Plate-Found': str(bool(result.plate_mask.any())).lower(),
        'Cache-Control': 'no-cache',
    })
