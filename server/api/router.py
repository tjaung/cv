from fastapi import APIRouter, Response

from .features import get_features, get_feature_histograms
from .api import get_image, get_image_sets, get_images, get_test_ground_truth
from .preprocessing import get_preprocessing_steps, run_preprocessing_pipeline

api_router = APIRouter()

api_router.add_api_route('/features/', get_features, methods=['GET'], tags=['Features'])
api_router.add_api_route('/features/histograms/', get_feature_histograms, methods=['GET'], tags=['Features'])

api_router.add_api_route('/preprocessing/steps/', get_preprocessing_steps, methods=['GET'], tags=['Preprocessing'])
api_router.add_api_route(
    '/preprocessing/pipeline/', run_preprocessing_pipeline, methods=['GET'],
    tags=['Preprocessing'], response_class=Response,
    responses={200: {'content': {'image/png': {}}}},
)

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
