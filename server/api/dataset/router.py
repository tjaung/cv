from fastapi import APIRouter
from .handlers import get_image_sets, get_images, get_image, get_test_ground_truth

router = APIRouter()

router.add_api_route("/image_sets", get_image_sets, methods=["GET"])
router.add_api_route("/images/{image_set}/{split}", get_images, methods=["GET"])
router.add_api_route(
    "/images/{image_set}/{split}/{image_name:path}", get_image, methods=["GET"]
)
router.add_api_route(
    "/test/ground_truth/{image_set}/{defect}/{image_name}",
    get_test_ground_truth,
    methods=["GET"],
)
