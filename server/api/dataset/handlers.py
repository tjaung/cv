import base64
import mimetypes
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from ..shared.os_helpers import IMAGE_EXTENSIONS, child_folders, folder_path, image_path

def get_image_sets():
    """List dataset categories and their immediate child folders."""
    root = folder_path()
    return {"image_sets": {
        name: child_folders(folder_path(name)) for name in child_folders(root)
    }}


def get_images(image_set: str, split: str):
    """List image names relative to a split, including defect subfolders."""
    folder = folder_path(image_set, split)
    images = []
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_path(image_set, split, relative.as_posix())
            images.append(relative.as_posix())
    return {"images": sorted(images)}


def get_image(image_set: str, split: str, image_name: str):
    """Serve an image's original bytes and media type."""
    return FileResponse(image_path(image_set, split, image_name))


def image_data(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def get_test_ground_truth(image_set: str, defect: str, image_name: str):
    """Return a test image and its matching mask in one JSON response."""
    test = image_path(image_set, "test", defect, image_name)
    ground_truth = None
    if defect != "good":
        mask_name = f"{Path(image_name).stem}_mask.png"
        try:
            mask = image_path(image_set, "ground_truth", defect, mask_name)
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(404, "Corresponding ground-truth image not found") from exc
            raise
        ground_truth = {"name": f"{defect}/{mask.name}", "data_url": image_data(mask)}
    return {
        "test": {"name": f"{defect}/{test.name}", "data_url": image_data(test)},
        "ground_truth": ground_truth,
    }
