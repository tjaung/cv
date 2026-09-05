from pathlib import Path
import cv2
import matplotlib.pyplot as plt

if __package__:
    from .helpers import get_image
else:
    from helpers import get_image

def get_color_histogram(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def get_grayscale_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def plot_histograms(histograms, title, columns=6):
    if not histograms:
        raise ValueError("No image histograms to plot")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    columns = min(columns, len(histograms))
    rows = (len(histograms) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 3, rows * 2.3),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    figure.suptitle(title)
    figure.supxlabel("Bins")
    figure.supylabel("Normalized frequency")
    for axis, (name, hist) in zip(axes.flat, histograms):
        axis.plot(hist, linewidth=0.8)
        axis.set_title(Path(name).name, fontsize=9)
        axis.set_xlim(0, len(hist) - 1)
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[len(histograms):]:
        axis.set_visible(False)
    return figure

def load_histograms(images_path):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    color_histograms = []
    grayscale_histograms = []
    for image in sorted(Path(images_path).iterdir()):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue
        img = get_image(str(image))
        color_histograms.append((image.name, get_color_histogram(img)))
        grayscale_histograms.append((image.name, get_grayscale_histogram(img)))

    return color_histograms, grayscale_histograms
