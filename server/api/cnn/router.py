from fastapi import APIRouter
from .handlers import get_cnn, train_cnn, test_cnn, cnn_weights, cnn_predictions, cnn_inspect, save_cnn

router = APIRouter()

router.add_api_route('/cnn/', get_cnn, methods=['GET'], tags=['CNN'])
router.add_api_route('/cnn/train/', train_cnn, methods=['POST'], tags=['CNN'])
router.add_api_route('/cnn/test/', test_cnn, methods=['POST'], tags=['CNN'])
router.add_api_route('/cnn/weights/', cnn_weights, methods=['GET'], tags=['CNN'])
router.add_api_route('/cnn/predictions/', cnn_predictions, methods=['GET'], tags=['CNN'])
router.add_api_route('/cnn/inspect/', cnn_inspect, methods=['GET'], tags=['CNN'])
router.add_api_route('/cnn/save/', save_cnn, methods=['POST'], tags=['CNN'])
