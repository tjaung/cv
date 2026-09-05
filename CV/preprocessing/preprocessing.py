import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .inpainting import glare_threshold
    from .surrounding_color_fill import fill_surrounding_color
    from .boundary_blending import blend_glare_boundary
    from .helpers import get_image
    from .luminance_correction import adjust_luminance_to_middle_by_percentage
    from .grayscale import to_grayscale
    from .sobel_edge_bold import apply_sobel_edge_bold
    from .gaussian_blur import apply_gaussian_blur
    from .isodata import apply_isodata
    from .morphological_cleanup import morphological_cleanup
    from .plate_mask import extract_plate_mask
    from .apply_mask import apply_plate_mask
    from .clahe import apply_clahe
    from .canny_edge import apply_canny_edge_detection
else:
    from inpainting import glare_threshold
    from surrounding_color_fill import fill_surrounding_color
    from boundary_blending import blend_glare_boundary
    from helpers import get_image
    from luminance_correction import adjust_luminance_to_middle_by_percentage
    from grayscale import to_grayscale
    from sobel_edge_bold import apply_sobel_edge_bold
    from gaussian_blur import apply_gaussian_blur
    from isodata import apply_isodata
    from morphological_cleanup import morphological_cleanup
    from plate_mask import extract_plate_mask
    from apply_mask import apply_plate_mask
    from clahe import apply_clahe
    from canny_edge import apply_canny_edge_detection


@dataclass
class PreprocessingResult:
    normalized: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edge_bold: np.ndarray
    threshold: float
    threshold_mask: np.ndarray
    cleaned_mask: np.ndarray
    plate_mask: np.ndarray
    plate: np.ndarray
    plate_gray: np.ndarray
    glare_mask: np.ndarray
    color_filled_plate: np.ndarray
    blended_plate: np.ndarray
    blurred_plate: np.ndarray
    clahe_plate: np.ndarray
    segmented_edges: np.ndarray


def preprocess_plate(image, luminance_percentage=10, blur_size=5,
                     fill_holes=True, allow_border_touching=True, sobel_strength=1.0, threshold_offset=20,
                     glare_cutoff=220, surrounding_radius=5, run_glare_fill=True, blend_width=3):

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError('Expected a nonempty 8-bit BGR image')

    normalized = adjust_luminance_to_middle_by_percentage(image, luminance_percentage)
    gray = to_grayscale(image)
    blurred = apply_gaussian_blur(gray, blur_size)
    edge_bold = apply_sobel_edge_bold(blurred, sobel_strength)
    threshold_mask, threshold = apply_isodata(edge_bold, threshold_offset)
    cleaned_mask = morphological_cleanup(threshold_mask, fill_holes=fill_holes)
    plate_mask = extract_plate_mask(cleaned_mask, fill_holes, allow_border_touching)
    plate = apply_plate_mask(normalized, plate_mask)
    plate_gray = to_grayscale(plate)
    glare_mask = glare_threshold(plate_gray, plate_mask, glare_cutoff)
    color_filled_plate = (fill_surrounding_color(plate, glare_mask, plate_mask, surrounding_radius)
                          if run_glare_fill else plate.copy())
    blended_plate = blend_glare_boundary(plate, color_filled_plate, glare_mask, blend_width)
    blurred_plate = apply_gaussian_blur(to_grayscale(blended_plate), blur_size)
    blurred_plate[plate_mask == 0] = 0
    clahe_plate = apply_clahe(blurred_plate)
    clahe_plate[plate_mask == 0] = 0
    segmented_edges = apply_canny_edge_detection(clahe_plate)

    return PreprocessingResult(normalized, gray, blurred, edge_bold, threshold, threshold_mask,
                               cleaned_mask, plate_mask, plate,
                               plate_gray, glare_mask, color_filled_plate, blended_plate,
                               blurred_plate, clahe_plate, segmented_edges)


def main():
    default_image = Path(__file__).resolve().parents[2] / 'server/anomaly_dataset/metal_plate/train/good/000.png'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, default=default_image)
    parser.add_argument('--luminance-percentage', type=float, default=10)
    parser.add_argument('--blur-size', type=int, default=5)
    parser.add_argument('--sobel-strength', type=float, default=1.0)
    parser.add_argument('--threshold-offset', type=float, default=20)
    parser.add_argument('--glare-cutoff', type=float, default=220)
    parser.add_argument('--surrounding-radius', type=int, default=5)
    parser.add_argument('--blend-width', type=int, default=3)
    parser.add_argument('--keep-holes', action='store_true')
    parser.add_argument('--allow-border-touching', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output-dir', type=Path, help='Save stages and preview instead of opening a window')
    args = parser.parse_args()
    image = get_image(str(args.image))
    result = preprocess_plate(image, args.luminance_percentage, args.blur_size,
                              not args.keep_holes, args.allow_border_touching, args.sobel_strength, args.threshold_offset,
                              args.glare_cutoff, args.surrounding_radius, blend_width=args.blend_width)
    if not result.plate_mask.any():
        print('No enclosed plate found. Check contrast, morphology, or --allow-border-touching.')

    stages = [('Raw image', image), ('Normalized', result.normalized),
              ('Grayscale', result.gray), ('Blur', result.blurred), ('Edge darkening', result.edge_bold),
              ('ISODATA', result.threshold_mask),
              ('Morphological cleanup', result.cleaned_mask), ('Plate mask', result.plate_mask),
              ('Segmented plate', result.plate),
              ('Plate grayscale', result.plate_gray), ('Glare threshold', result.glare_mask),
              ('Surrounding color fill', result.color_filled_plate), ('Boundary blend', result.blended_plate),
              ('Plate blur', result.blurred_plate), ('CLAHE', result.clahe_plate), ('Canny edges', result.segmented_edges)]
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 4, figsize=(16, 16), layout='constrained')
    figure.suptitle(args.image.name)
    for axis, (title, data) in zip(axes.flat, stages):
        display = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) if data.ndim == 3 else data
        axis.imshow(display, cmap='gray', vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis('off')
    for axis in list(axes.flat)[len(stages):]:
        axis.axis('off')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for title, data in stages:
            destination = args.output_dir / f'{args.image.stem}_{title.lower().replace(" ", "_")}.png'
            if not cv2.imwrite(str(destination), data):
                raise OSError(f'Could not write {destination}')
        figure.savefig(args.output_dir / f'{args.image.stem}_preview.png', dpi=120)
        plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
