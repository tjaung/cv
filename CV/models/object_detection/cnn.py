"""Dataset access for CNN experiments."""

from pathlib import Path

from ..data_shuffle import ImageSample, ImageStore


def get_train_test_data(
    root: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[ImageSample, ...], tuple[ImageSample, ...]]:
    """Return labeled training/test records using the shared classifier sampler.

    ``root`` is the class dataset directory (e.g. anomaly_dataset/metal_plate)
    containing train/ and test/. Both folders are pooled before splitting.
    Records contain path, label, original_split, and sha256; pixels are not
    loaded or preprocessed. Matching data and seed give the classifier split.
    """
    split = ImageStore(root).sample(test_size=test_size, seed=seed)
    return split.train, split.test
