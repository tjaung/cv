from pathlib import Path

from fastapi import HTTPException

DATASET_ROOT = Path(__file__).resolve().parents[2] / "anomaly_dataset"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def dataset_path(*parts: str) -> Path:
    root = DATASET_ROOT.resolve()
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    return path


def folder_path(*parts: str) -> Path:
    path = dataset_path(*parts)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    return path


def image_path(*parts: str) -> Path:
    path = dataset_path(*parts)
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")
    return path


def child_folders(path: Path) -> list[str]:
    return sorted(
        child.name for child in path.iterdir()
        if not child.name.startswith(".") and child.is_dir()
        and child.resolve().is_relative_to(DATASET_ROOT.resolve())
    )
