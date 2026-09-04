from fastapi import APIRouter

from .api import get_image, get_image_sets, get_images, get_test_ground_truth

api_router = APIRouter()

api_router.add_api_route("/image_sets", get_image_sets, methods=["GET"])
api_router.add_api_route("/images/{image_set}/{split}", get_images, methods=["GET"])
api_router.add_api_route(
    "/images/{image_set}/{split}/{image_name:path}", get_image, methods=["GET"]
)
api_router.add_api_route(
    "/test/ground_truth/{image_set}/{defect}/{image_name}",
    get_test_ground_truth,
    methods=["GET"],
)
