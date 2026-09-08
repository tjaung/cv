from fastapi import APIRouter
from .one_class_svm import train_svm_grid
from .models import get_pca_model, test_pca, evaluate_pca, get_model_projection, get_models, get_model_job, train_pca_sweep, evaluate_all_pca, train_pca, get_pca_features, download_pca_features
from .model_lifecycle import clear_models, retrain_all_models, download_history

router = APIRouter()

router.add_api_route('/models/one_class_svm/grid/', train_svm_grid, methods=['POST'], tags=['Models'])
router.add_api_route('/models/one_class_svm/', get_pca_model, methods=['GET'], tags=['Models'])
router.add_api_route('/models/one_class_svm/test/', test_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/one_class_svm/evaluate/', evaluate_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/clear/', clear_models, methods=['POST'], tags=['Models'])
router.add_api_route('/models/retrain_all/', retrain_all_models, methods=['POST'], tags=['Models'])
router.add_api_route('/models/history/{model_id}', download_history, methods=['GET'], tags=['Models'])
router.add_api_route('/models/projection/', get_model_projection, methods=['GET'], tags=['Models'])
router.add_api_route('/models/', get_models, methods=['GET'], tags=['Models'])
router.add_api_route('/models/jobs/{job_id}', get_model_job, methods=['GET'], tags=['Models'])
router.add_api_route('/models/pca/', get_pca_model, methods=['GET'], tags=['Models'])
router.add_api_route('/models/pca/sweep/', train_pca_sweep, methods=['POST'], tags=['Models'])
router.add_api_route('/models/pca/evaluate_all/', evaluate_all_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/pca/train/', train_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/pca/test/', test_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/pca/evaluate/', evaluate_pca, methods=['POST'], tags=['Models'])
router.add_api_route('/models/pca/features/', get_pca_features, methods=['GET'], tags=['Models'])
router.add_api_route('/models/pca/download/', download_pca_features, methods=['GET'], tags=['Models'])
