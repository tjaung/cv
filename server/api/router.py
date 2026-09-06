from .model_lifecycle import clear_models, retrain_all_models, download_history
from .one_class_svm import train_svm_grid
from fastapi import APIRouter, Response

from .models import get_model_projection, train_pca_sweep, evaluate_all_pca, get_models, get_pca_model, train_pca, test_pca, evaluate_pca, get_model_job, get_pca_features, download_pca_features
from .features import get_features, get_feature_histograms
from .api import get_image, get_image_sets, get_images, get_test_ground_truth
from .preprocessing import get_preprocessing_steps, run_preprocessing_pipeline

api_router = APIRouter()

api_router.add_api_route('/models/one_class_svm/grid/', train_svm_grid, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/one_class_svm/', get_pca_model, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/one_class_svm/test/', test_pca, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/one_class_svm/evaluate/', evaluate_pca, methods=['POST'], tags=['Models'])

api_router.add_api_route('/models/clear/', clear_models, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/retrain_all/', retrain_all_models, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/history/{model_id}', download_history, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/projection/', get_model_projection, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/', get_models, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/jobs/{job_id}', get_model_job, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/pca/', get_pca_model, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/pca/sweep/', train_pca_sweep, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/pca/evaluate_all/', evaluate_all_pca, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/pca/train/', train_pca, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/pca/test/', test_pca, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/pca/evaluate/', evaluate_pca, methods=['POST'], tags=['Models'])
api_router.add_api_route('/models/pca/features/', get_pca_features, methods=['GET'], tags=['Models'])
api_router.add_api_route('/models/pca/download/', download_pca_features, methods=['GET'], tags=['Models'])

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
